"""Flux Klein image t2i quant — owns hyperparams and denoise loop."""

from __future__ import annotations

import random

import torch
from flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise,
    denoise_cached,
    denoise_cfg,
    encode_image_refs,
    get_schedule,
    scatter_ids,
)
from reqm.overrides_ext import override

from infusers.model.klein import KleinModel
from infusers.quant.api.image_base import ImageOutput, ImageQuant, chw_float01_to_pil


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
    def forward(
        self,
        prompt: str,
        seed: int | None = None,
        resolution: list[int] | None = None,
        cond_images: list[torch.Tensor] | None = None,
    ) -> ImageOutput:
        height, width = resolution or self.resolution
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        device = self.model.device
        text_encoder = self.model.text_encoder
        flow = self.model.flow
        ae = self.model.ae
        model_info = self.model.model_info

        ref_tokens = None
        ref_ids = None
        if cond_images:
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

        timesteps = get_schedule(self.num_steps, x.shape[1])
        if model_info["guidance_distilled"]:
            denoise_fn = (
                denoise_cached
                if (model_info.get("use_kv_cache") and ref_tokens is not None)
                else denoise
            )
            x = denoise_fn(
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
            x = denoise_cfg(
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

        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        x = ae.decode(x).float()

        if self.cpu_offload:
            self.model.flow = flow.cpu()
            torch.cuda.empty_cache()
            self.model.text_encoder = text_encoder.to(device)

        x = x.clamp(-1, 1)
        chw = ((x[0] + 1.0) / 2.0).float()
        return ImageOutput(image=chw)

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
