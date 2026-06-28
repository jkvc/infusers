"""Flux Klein image t2i quant — owns hyperparams and denoise loop."""

from __future__ import annotations

import random
from collections.abc import Iterator

import torch
from flux2.sampling import (
    Flux2,
    batched_prc_img,
    batched_prc_txt,
    encode_image_refs,
    get_schedule,
    scatter_ids,
)
from reqm.overrides_ext import override
from torch import Tensor

from infusers.model.klein import KleinModel
from infusers.quant.api.image_base import (
    ImageIntermediateEvent,
    ImageOutput,
    ImageQuant,
    chw_float01_to_pil,
)


def _iter_denoise(
    model: Flux2,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: list[float],
    guidance: float,
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
) -> Iterator[tuple[int, int]]:
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    total = max(len(timesteps) - 1, 0)
    for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        yield step_idx + 1, total
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        img_input = img
        img_input_ids = img_ids
        if img_cond_seq is not None:
            assert img_cond_seq_ids is not None
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)
        pred = model(
            x=img_input,
            x_ids=img_input_ids,
            timesteps=t_vec,
            ctx=txt,
            ctx_ids=txt_ids,
            guidance=guidance_vec,
        )
        if img_input_ids is not None:
            pred = pred[:, : img.shape[1]]
        img = img + (t_prev - t_curr) * pred
    return img


def _iter_denoise_cached(
    model: Flux2,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: list[float],
    guidance: float,
    img_cond_seq: Tensor,
    img_cond_seq_ids: Tensor,
) -> Iterator[tuple[int, int]]:
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    total = max(len(timesteps) - 1, 0)
    kv_cache = None
    for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        yield step_idx + 1, total
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        if step_idx == 0:
            pred, kv_cache = model.forward_kv_extract(
                x=img,
                x_ids=img_ids,
                timesteps=t_vec,
                ctx=txt,
                ctx_ids=txt_ids,
                guidance=guidance_vec,
                x_seq_concat=img_cond_seq,
                x_seq_concat_ids=img_cond_seq_ids,
            )
        else:
            pred = model.forward_kv_cached(
                x=img,
                x_ids=img_ids,
                timesteps=t_vec,
                ctx=txt,
                ctx_ids=txt_ids,
                guidance=guidance_vec,
                kv_cache=kv_cache,
            )
        img = img + (t_prev - t_curr) * pred
    return img


def _iter_denoise_cfg(
    model: Flux2,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: list[float],
    guidance: float,
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
) -> Iterator[tuple[int, int]]:
    img = torch.cat([img, img], dim=0)
    img_ids = torch.cat([img_ids, img_ids], dim=0)
    if img_cond_seq is not None:
        assert img_cond_seq_ids is not None
        img_cond_seq = torch.cat([img_cond_seq, img_cond_seq], dim=0)
        img_cond_seq_ids = torch.cat([img_cond_seq_ids, img_cond_seq_ids], dim=0)

    total = max(len(timesteps) - 1, 0)
    for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        yield step_idx + 1, total
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        img_input = img
        img_input_ids = img_ids
        if img_cond_seq is not None:
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)
        pred = model(
            x=img_input,
            x_ids=img_input_ids,
            timesteps=t_vec,
            ctx=txt,
            ctx_ids=txt_ids,
            guidance=None,
        )
        if img_cond_seq is not None:
            pred = pred[:, : img.shape[1]]
        pred_uncond, pred_cond = pred.chunk(2)
        pred = pred_uncond + guidance * (pred_cond - pred_uncond)
        pred = torch.cat([pred, pred], dim=0)
        img = img + (t_prev - t_curr) * pred
    return img.chunk(2)[0]


class FluxImageQuant(ImageQuant):
    def __init__(
        self,
        model: KleinModel,
        num_steps: int,
        guidance: float,
        resolution: list[int],
        cpu_offload: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        self.guidance = guidance
        self.resolution = resolution
        self.cpu_offload = cpu_offload
        if cpu_offload:
            self.model.flow = self.model.flow.cpu()

    @torch.no_grad()
    @override
    def forward_gen(
        self,
        prompt: str,
        seed: int | None = None,
        resolution: list[int] | None = None,
        cond_images: list[torch.Tensor] | None = None,
        num_steps: int | None = None,
    ) -> Iterator[ImageIntermediateEvent | ImageOutput]:
        height, width = resolution or self.resolution
        steps = self.num_steps if num_steps is None else num_steps
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        device = self.model.device
        text_encoder = self.model.text_encoder
        flow = self.model.flow
        ae = self.model.ae
        model_info = self.model.model_info

        yield ImageIntermediateEvent(message="encode prompt")

        ref_tokens = None
        ref_ids = None
        if cond_images:
            yield ImageIntermediateEvent(message="encode reference images")
            ref_pil = [chw_float01_to_pil(t) for t in cond_images]
            ref_tokens, ref_ids = encode_image_refs(ae, ref_pil)

        if model_info["guidance_distilled"]:
            ctx = text_encoder([prompt]).to(torch.bfloat16)
        else:
            ctx_empty = text_encoder([""]).to(torch.bfloat16)
            ctx_prompt = text_encoder([prompt]).to(torch.bfloat16)
            ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
        ctx, ctx_ids = batched_prc_txt(ctx)

        if self.cpu_offload:
            text_encoder = text_encoder.cpu()
            torch.cuda.empty_cache()
            flow = flow.to(device)

        shape = (1, 128, height // 16, width // 16)
        generator = torch.Generator(device=device).manual_seed(seed)
        randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device=device)
        x, x_ids = batched_prc_img(randn)

        timesteps = get_schedule(steps, x.shape[1])
        total_steps = max(len(timesteps) - 1, 0)
        yield ImageIntermediateEvent(message=f"denoise begin ({total_steps} steps)")

        if model_info["guidance_distilled"]:
            if model_info.get("use_kv_cache") and ref_tokens is not None:
                denoise_iter = _iter_denoise_cached(
                    flow,
                    x,
                    x_ids,
                    ctx,
                    ctx_ids,
                    timesteps=timesteps,
                    guidance=self.guidance,
                    img_cond_seq=ref_tokens,
                    img_cond_seq_ids=ref_ids,
                )
            else:
                denoise_iter = _iter_denoise(
                    flow,
                    x,
                    x_ids,
                    ctx,
                    ctx_ids,
                    timesteps=timesteps,
                    guidance=self.guidance,
                    img_cond_seq=ref_tokens,
                    img_cond_seq_ids=ref_ids,
                )
        else:
            denoise_iter = _iter_denoise_cfg(
                flow,
                x,
                x_ids,
                ctx,
                ctx_ids,
                timesteps=timesteps,
                guidance=self.guidance,
                img_cond_seq=ref_tokens,
                img_cond_seq_ids=ref_ids,
            )

        try:
            while True:
                current, total = next(denoise_iter)
                yield ImageIntermediateEvent(message=f"denoise step {current}/{total}")
        except StopIteration as exc:
            if exc.value is None:
                raise RuntimeError("denoise iterator did not return a tensor") from exc
            x = exc.value

        yield ImageIntermediateEvent(message="decode image")

        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        x = ae.decode(x).float()

        if self.cpu_offload:
            self.model.flow = flow.cpu()
            torch.cuda.empty_cache()
            self.model.text_encoder = text_encoder.to(device)

        x = x.clamp(-1, 1)
        chw = ((x[0] + 1.0) / 2.0).float()
        yield ImageOutput(message="image ready", image=chw)

    @override
    def dummy_inputs(self) -> list[dict[str, object]]:
        return [
            {
                "prompt": "solid gray",
                "seed": 0,
                "resolution": [512, 512],
                "cond_images": None,
            }
        ]
