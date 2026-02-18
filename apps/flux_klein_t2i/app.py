"""
FLUX.2 Klein 9B text-to-image inference service.

Deploy:  fal deploy flux-klein-t2i
Test:    fal run flux-klein-t2i
"""

import os
from typing import Literal

import fal
from fal.toolkit.image import Image, ImageSize, ImageSizeInput, get_image_size
from pydantic import BaseModel, Field


class Input(BaseModel):
    prompt: str = Field(
        description="Text prompt to generate an image from.",
        examples=[
            "A cat holding a sign that says hello world",
        ],
    )
    image_size: ImageSizeInput = Field(
        default=ImageSize(width=1024, height=1024),
        description="The size of the generated image.",
    )
    num_inference_steps: int = Field(
        default=50,
        description="Number of denoising steps. More steps = higher quality but slower.",
        ge=1,
        le=100,
    )
    guidance_scale: float = Field(
        default=4.0,
        description="How closely the model follows the prompt. Higher = more faithful.",
        ge=0.0,
        le=10.0,
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility. Leave empty for random.",
    )
    num_images: int = Field(
        default=1,
        description="Number of images to generate.",
        ge=1,
        le=4,
    )
    output_format: Literal["jpeg", "png"] = Field(
        default="jpeg",
        description="Output image format.",
    )


class Output(BaseModel):
    images: list[Image] = Field(description="Generated images.")
    seed: int = Field(description="Seed used for generation.")
    prompt: str = Field(description="The prompt that was used.")


class FluxKleinT2I(fal.App):
    machine_type = "GPU-A100"
    keep_alive = 300
    min_concurrency = 0
    max_concurrency = 1

    requirements = [
        "torch==2.6.0",
        "accelerate==1.6.0",
        "transformers>=4.51.0",
        "git+https://github.com/huggingface/diffusers.git@main",
        "sentencepiece==0.2.0",
        "protobuf>=4.25.0",
        "hf_transfer==0.1.9",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu124",
    ]

    def setup(self):
        import torch
        from diffusers import Flux2KleinPipeline

        # HF_TOKEN is set via `fal secret set HF_TOKEN <value>`
        self.pipe = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-9B",
            torch_dtype=torch.bfloat16,
            token=os.environ.get("HF_TOKEN"),
        )
        self.pipe.to("cuda")

    @fal.endpoint("/")
    async def generate(self, input: Input) -> Output:
        import torch

        image_size = get_image_size(input.image_size)
        seed = input.seed if input.seed is not None else torch.seed()
        generator = torch.Generator("cuda").manual_seed(seed)

        images = self.pipe(
            prompt=input.prompt,
            height=image_size.height,
            width=image_size.width,
            num_inference_steps=input.num_inference_steps,
            guidance_scale=input.guidance_scale,
            num_images_per_prompt=input.num_images,
            generator=generator,
        ).images

        return Output(
            images=[Image.from_pil(img, input.output_format) for img in images],
            seed=seed,
            prompt=input.prompt,
        )
