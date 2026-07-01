"""Flux Klein cylindrical panorama quant — multi-slice tiled denoise."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from einops import rearrange, repeat
from flux2.sampling import (
    Flux2,
    batched_prc_img,
    batched_prc_txt,
    encode_image_refs,
    get_schedule,
)
from reqm.overrides_ext import override
from torch import Tensor

from infusers.model.klein import KleinModel
from infusers.quant.api.image_base import chw_float01_to_pil
from infusers.quant.api.pano_base import (
    PanoramaIntermediateEvent,
    PanoramaOutput,
    PanoramaQuant,
)

_LATENT_SCALE = 16


@dataclass(frozen=True)
class CanvasDims:
    slice_height: int
    slice_width: int
    overlap_pixels: int
    overlap_latent: int
    output_height: int
    output_width: int
    latent_full_height: int
    latent_full_width: int
    slice_height_latent: int
    slice_width_latent: int


def validate_inputs(
    prompts: Sequence[str],
    resolution: list[int],
    overlap_pixels: int,
    pano_direction: str,
    cond_images: list[torch.Tensor] | list[list[torch.Tensor]] | None,
) -> None:
    if not prompts:
        raise ValueError("prompts must not be empty")
    if pano_direction not in ("horizontal", "vertical"):
        raise ValueError(
            f"pano_direction must be 'horizontal' or 'vertical', got {pano_direction!r}"
        )
    if len(resolution) != 2:
        raise ValueError(f"resolution must be [height, width], got {resolution!r}")
    height, width = resolution
    if height % _LATENT_SCALE != 0 or width % _LATENT_SCALE != 0:
        raise ValueError(
            f"resolution height and width must be divisible by {_LATENT_SCALE}, "
            f"got {resolution!r}"
        )
    if overlap_pixels <= 0:
        raise ValueError(f"overlap_pixels must be > 0, got {overlap_pixels}")
    if overlap_pixels % _LATENT_SCALE != 0:
        raise ValueError(
            f"overlap_pixels must be divisible by {_LATENT_SCALE}, got {overlap_pixels}"
        )
    overlap_latent = overlap_pixels // _LATENT_SCALE
    if overlap_latent < 1:
        raise ValueError(
            f"overlap_pixels // {_LATENT_SCALE} must be >= 1 for cylindrical pano, "
            f"got overlap_pixels={overlap_pixels}"
        )
    pano_axis = width if pano_direction == "horizontal" else height
    if overlap_pixels >= pano_axis:
        raise ValueError(
            f"overlap_pixels ({overlap_pixels}) must be < slice "
            f"{'width' if pano_direction == 'horizontal' else 'height'} ({pano_axis})"
        )
    if cond_images is not None and cond_images and isinstance(cond_images[0], list):
        per_slice: list[list[torch.Tensor]] = cond_images  # type: ignore[assignment]
        if len(per_slice) != len(prompts):
            raise ValueError(
                f"cond_images outer list length ({len(per_slice)}) must match "
                f"len(prompts) ({len(prompts)})"
            )


def compute_canvas_dims(
    num_slices: int,
    slice_height: int,
    slice_width: int,
    overlap_pixels: int,
    pano_direction: str,
) -> CanvasDims:
    overlap_latent = overlap_pixels // _LATENT_SCALE
    if pano_direction == "horizontal":
        output_width = (slice_width - overlap_pixels) * num_slices
        output_height = slice_height
    else:
        output_width = slice_width
        output_height = (slice_height - overlap_pixels) * num_slices
    return CanvasDims(
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_pixels=overlap_pixels,
        overlap_latent=overlap_latent,
        output_height=output_height,
        output_width=output_width,
        latent_full_height=output_height // _LATENT_SCALE,
        latent_full_width=output_width // _LATENT_SCALE,
        slice_height_latent=slice_height // _LATENT_SCALE,
        slice_width_latent=slice_width // _LATENT_SCALE,
    )


def create_slice_ids(
    height: int,
    width: int,
    device: torch.device,
    t_coord: int = 0,
) -> Tensor:
    x_ids = torch.zeros(height * width, 4, dtype=torch.int64, device=device)
    idx = 0
    for hi in range(height):
        for wi in range(width):
            x_ids[idx, 0] = t_coord
            x_ids[idx, 1] = hi
            x_ids[idx, 2] = wi
            x_ids[idx, 3] = 0
            idx += 1
    return x_ids.unsqueeze(0)


def build_lerp_mask_slice(
    slice_height_latent: int,
    slice_width_latent: int,
    overlap_latent: int,
    pano_direction: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    h = slice_height_latent
    w = slice_width_latent
    if pano_direction == "horizontal":
        ramp_left = torch.linspace(0, 1, overlap_latent, device=device, dtype=dtype)
        ramp_left = repeat(ramp_left, "w -> 1 1 h w", h=h)
        ramp_right = torch.linspace(1, 0, overlap_latent, device=device, dtype=dtype)
        ramp_right = repeat(ramp_right, "w -> 1 1 h w", h=h)
        center_w = w - 2 * overlap_latent
        center = torch.ones(1, 1, h, center_w, device=device, dtype=dtype)
        return torch.cat([ramp_left, center, ramp_right], dim=-1)
    ramp_top = torch.linspace(0, 1, overlap_latent, device=device, dtype=dtype)
    ramp_top = repeat(ramp_top, "h -> 1 1 h w", w=w)
    ramp_bottom = torch.linspace(1, 0, overlap_latent, device=device, dtype=dtype)
    ramp_bottom = repeat(ramp_bottom, "h -> 1 1 h w", w=w)
    center_h = h - 2 * overlap_latent
    center = torch.ones(1, 1, center_h, w, device=device, dtype=dtype)
    return torch.cat([ramp_top, center, ramp_bottom], dim=-2)


def fold_wraparound_horizontal(
    pred_wraparound: Tensor, full_width: int, overlap_latent: int
) -> Tensor:
    pred_final = pred_wraparound[:, :, :, :full_width].clone()
    pred_final[:, :, :, :overlap_latent] += pred_wraparound[:, :, :, -overlap_latent:]
    return pred_final


def fold_wraparound_vertical(
    pred_wraparound: Tensor, full_height: int, overlap_latent: int
) -> Tensor:
    pred_final = pred_wraparound[:, :, :full_height, :].clone()
    pred_final[:, :, :overlap_latent, :] += pred_wraparound[:, :, -overlap_latent:, :]
    return pred_final


def vae_decode_crop_slices(
    *,
    pano_direction: str,
    pad_latent: int,
    pad_apparent: int,
    true_apparent_width: int,
    true_apparent_height: int,
) -> tuple[slice, slice]:
    if pano_direction == "horizontal":
        return (
            slice(None),
            slice(pad_apparent, pad_apparent + true_apparent_width),
        )
    return (
        slice(pad_apparent, pad_apparent + true_apparent_height),
        slice(None),
    )


def _crop_axis_wrap(
    spatial: Tensor, start: int, extent: int, full_extent: int, axis: int
) -> Tensor:
    end = start + extent
    if end <= full_extent:
        return spatial.narrow(axis, start, extent)
    part1 = spatial.narrow(axis, start, full_extent - start)
    part2 = spatial.narrow(axis, 0, end - full_extent)
    return torch.cat([part1, part2], dim=axis)


def _normalize_cond_images(
    cond_images: list[torch.Tensor] | list[list[torch.Tensor]] | None,
    num_slices: int,
) -> list[list[torch.Tensor] | None]:
    if cond_images is None:
        return [None] * num_slices
    if not cond_images:
        return [None] * num_slices
    if isinstance(cond_images[0], torch.Tensor):
        shared: list[torch.Tensor] = cond_images  # type: ignore[assignment]
        return [shared] * num_slices
    return cond_images  # type: ignore[return-value]


def _encode_slice_cond(
    ae: torch.nn.Module,
    cond_tensors: list[torch.Tensor] | None,
    cache: dict[tuple[int, ...], tuple[Tensor, Tensor]],
) -> tuple[Tensor | None, Tensor | None]:
    if not cond_tensors:
        return None, None
    key = tuple(id(t) for t in cond_tensors)
    if key not in cache:
        ref_pil = [chw_float01_to_pil(t) for t in cond_tensors]
        cache[key] = encode_image_refs(ae, ref_pil)
    return cache[key]


def _flow_slice_pred(
    flow: Flux2,
    slice_spatial: Tensor,
    t_vec: Tensor,
    ctx: Tensor,
    ctx_ids: Tensor,
    guidance: float,
    ref_tokens: Tensor | None,
    ref_ids: Tensor | None,
) -> Tensor:
    slice_packed, slice_ids = batched_prc_img(slice_spatial)
    noisy_len = slice_packed.shape[1]
    img_input = slice_packed
    img_input_ids = slice_ids
    if ref_tokens is not None:
        assert ref_ids is not None
        img_input = torch.cat((img_input, ref_tokens), dim=1)
        img_input_ids = torch.cat((img_input_ids, ref_ids), dim=1)
    guidance_vec = torch.full(
        (slice_packed.shape[0],),
        guidance,
        device=slice_packed.device,
        dtype=slice_packed.dtype,
    )
    pred = flow(
        x=img_input,
        x_ids=img_input_ids,
        timesteps=t_vec,
        ctx=ctx,
        ctx_ids=ctx_ids,
        guidance=guidance_vec,
    )
    pred = pred[:, :noisy_len]
    _, _, h, w = slice_spatial.shape
    return rearrange(pred, "b (h w) c -> b c h w", h=h, w=w)


def _pred_noise(
    flow: Flux2,
    img: Tensor,
    t_vec: Tensor,
    dims: CanvasDims,
    pano_direction: str,
    num_slices: int,
    ctx: Tensor,
    ctx_ids: Tensor,
    guidance: float,
    slice_ref_tokens: list[tuple[Tensor | None, Tensor | None]],
) -> Tensor:
    img_spatial = rearrange(
        img,
        "b (h w) c -> b c h w",
        h=dims.latent_full_height,
        w=dims.latent_full_width,
    )
    overlap = dims.overlap_latent
    h_lat = dims.slice_height_latent
    w_lat = dims.slice_width_latent
    lerp_mask = build_lerp_mask_slice(
        h_lat,
        w_lat,
        overlap,
        pano_direction,
        device=img.device,
        dtype=img.dtype,
    )

    if pano_direction == "horizontal":
        wrap_w = dims.latent_full_width + overlap
        pred_wraparound = torch.zeros(
            1,
            img_spatial.shape[1],
            dims.latent_full_height,
            wrap_w,
            device=img.device,
            dtype=img.dtype,
        )
        step = w_lat - overlap
        for slice_idx in range(num_slices):
            left = slice_idx * step
            slice_spatial = _crop_axis_wrap(
                img_spatial, left, w_lat, dims.latent_full_width, axis=3
            )
            ref_tokens, ref_ids = slice_ref_tokens[slice_idx]
            slice_pred = _flow_slice_pred(
                flow,
                slice_spatial,
                t_vec,
                ctx[slice_idx : slice_idx + 1],
                ctx_ids[slice_idx : slice_idx + 1],
                guidance,
                ref_tokens,
                ref_ids,
            )
            pred_wraparound[:, :, :, left : left + w_lat] += slice_pred * lerp_mask
        pred_final = fold_wraparound_horizontal(pred_wraparound, dims.latent_full_width, overlap)
    else:
        wrap_h = dims.latent_full_height + overlap
        pred_wraparound = torch.zeros(
            1,
            img_spatial.shape[1],
            wrap_h,
            dims.latent_full_width,
            device=img.device,
            dtype=img.dtype,
        )
        step = h_lat - overlap
        for slice_idx in range(num_slices):
            top = slice_idx * step
            slice_spatial = _crop_axis_wrap(
                img_spatial, top, h_lat, dims.latent_full_height, axis=2
            )
            ref_tokens, ref_ids = slice_ref_tokens[slice_idx]
            slice_pred = _flow_slice_pred(
                flow,
                slice_spatial,
                t_vec,
                ctx[slice_idx : slice_idx + 1],
                ctx_ids[slice_idx : slice_idx + 1],
                guidance,
                ref_tokens,
                ref_ids,
            )
            pred_wraparound[:, :, top : top + h_lat, :] += slice_pred * lerp_mask
        pred_final = fold_wraparound_vertical(pred_wraparound, dims.latent_full_height, overlap)

    return rearrange(pred_final, "b c h w -> b (h w) c")


def _iter_pano_denoise(
    flow: Flux2,
    img: Tensor,
    dims: CanvasDims,
    pano_direction: str,
    num_slices: int,
    ctx: Tensor,
    ctx_ids: Tensor,
    timesteps: list[float],
    guidance: float,
    slice_ref_tokens: list[tuple[Tensor | None, Tensor | None]],
) -> Iterator[tuple[int, int]]:
    total = max(len(timesteps) - 1, 0)
    for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        yield step_idx + 1, total
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = _pred_noise(
            flow,
            img,
            t_vec,
            dims,
            pano_direction,
            num_slices,
            ctx,
            ctx_ids,
            guidance,
            slice_ref_tokens,
        )
        img = img + (t_prev - t_curr) * pred
    return img


def _decode_pano_latent(
    ae: torch.nn.Module,
    img_final: Tensor,
    dims: CanvasDims,
    pano_direction: str,
) -> Tensor:
    pad_latent = dims.overlap_latent
    if pad_latent == 0:
        return ae.decode(img_final).float()

    pad_apparent = pad_latent * _LATENT_SCALE
    if pano_direction == "horizontal":
        img_decode = torch.cat(
            [
                img_final[:, :, :, -pad_latent:],
                img_final,
                img_final[:, :, :, :pad_latent],
            ],
            dim=-1,
        )
    else:
        img_decode = torch.cat(
            [
                img_final[:, :, -pad_latent:, :],
                img_final,
                img_final[:, :, :pad_latent, :],
            ],
            dim=-2,
        )
    x = ae.decode(img_decode).float()
    h_slice, w_slice = vae_decode_crop_slices(
        pano_direction=pano_direction,
        pad_latent=pad_latent,
        pad_apparent=pad_apparent,
        true_apparent_width=dims.output_width,
        true_apparent_height=dims.output_height,
    )
    return x[:, :, h_slice, w_slice]


class FluxPanoramaQuant(PanoramaQuant):
    def __init__(
        self,
        model: KleinModel,
        num_steps: int,
        guidance: float,
        resolution: list[int],
        pano_direction: str = "horizontal",
        overlap_pixels: int = 256,
        cpu_offload: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        self.guidance = guidance
        self.resolution = resolution
        self.pano_direction = pano_direction
        self.overlap_pixels = overlap_pixels
        self.cpu_offload = cpu_offload
        if cpu_offload:
            raise NotImplementedError("cpu_offload is not supported for FluxPanoramaQuant yet")

    @torch.no_grad()
    @override
    def forward_gen(
        self,
        prompts: list[str],
        seed: int | None = None,
        resolution: list[int] | None = None,
        pano_direction: str | None = None,
        overlap_pixels: int | None = None,
        cond_images: list[torch.Tensor] | list[list[torch.Tensor]] | None = None,
        num_steps: int | None = None,
    ) -> Iterator[PanoramaIntermediateEvent | PanoramaOutput]:
        slice_resolution = resolution or self.resolution
        direction = pano_direction or self.pano_direction
        overlap = overlap_pixels if overlap_pixels is not None else self.overlap_pixels
        steps = self.num_steps if num_steps is None else num_steps
        num_slices = len(prompts)

        validate_inputs(prompts, slice_resolution, overlap, direction, cond_images)
        dims = compute_canvas_dims(
            num_slices,
            slice_resolution[0],
            slice_resolution[1],
            overlap,
            direction,
        )

        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        device = self.model.device
        text_encoder = self.model.text_encoder
        flow = self.model.flow
        ae = self.model.ae
        model_info = self.model.model_info
        if not model_info["guidance_distilled"]:
            raise NotImplementedError(
                "FluxPanoramaQuant MVP supports guidance-distilled Klein models only"
            )

        yield PanoramaIntermediateEvent(message="encode prompts")

        per_slice_cond = _normalize_cond_images(cond_images, num_slices)
        encode_cache: dict[tuple[int, ...], tuple[Tensor, Tensor]] = {}
        slice_ref_tokens: list[tuple[Tensor | None, Tensor | None]] = []
        if any(c is not None for c in per_slice_cond):
            yield PanoramaIntermediateEvent(message="encode reference images")
        for cond in per_slice_cond:
            slice_ref_tokens.append(_encode_slice_cond(ae, cond, encode_cache))

        ctx = text_encoder(prompts).to(torch.bfloat16)
        ctx, ctx_ids = batched_prc_txt(ctx)

        shape = (1, 128, dims.latent_full_height, dims.latent_full_width)
        generator = torch.Generator(device=device).manual_seed(seed)
        randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device=device)
        img, _img_ids = batched_prc_img(randn)

        timesteps = get_schedule(steps, img.shape[1])
        total_steps = max(len(timesteps) - 1, 0)
        yield PanoramaIntermediateEvent(
            message=f"denoise begin ({total_steps} steps, {num_slices} slices)"
        )

        denoise_iter = _iter_pano_denoise(
            flow,
            img,
            dims,
            direction,
            num_slices,
            ctx,
            ctx_ids,
            timesteps=timesteps,
            guidance=self.guidance,
            slice_ref_tokens=slice_ref_tokens,
        )
        try:
            while True:
                current, total = next(denoise_iter)
                yield PanoramaIntermediateEvent(message=f"denoise step {current}/{total}")
        except StopIteration as exc:
            if exc.value is None:
                raise RuntimeError("pano denoise iterator did not return a tensor") from exc
            img = exc.value

        yield PanoramaIntermediateEvent(message="decode image")

        img_spatial = rearrange(
            img,
            "b (h w) c -> b c h w",
            h=dims.latent_full_height,
            w=dims.latent_full_width,
        )
        decoded = _decode_pano_latent(ae, img_spatial, dims, direction)
        decoded = decoded.clamp(-1, 1)
        chw = ((decoded[0] + 1.0) / 2.0).float()

        yield PanoramaOutput(
            message="panorama ready",
            image=chw,
            direction=direction,
            slice_resolution=list(slice_resolution),
            output_size=[dims.output_height, dims.output_width],
            num_slices=num_slices,
            overlap_pixels=overlap,
        )

    @override
    def dummy_inputs(self) -> list[dict[str, object]]:
        return [
            {
                "prompts": ["desert dunes", "oasis palms"],
                "seed": 0,
                "resolution": [512, 1024],
                "pano_direction": "horizontal",
                "overlap_pixels": 256,
                "cond_images": None,
            }
        ]
