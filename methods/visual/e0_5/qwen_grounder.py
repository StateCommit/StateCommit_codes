"""Frozen Qwen2.5-VL adapter for the public E0.5 Grounder protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .grounder_protocol import build_grounder_prompt


def _public_change_crops(before: Path, after: Path) -> tuple["object", "object"]:
    """Return aligned, enlarged crops around RGB differences only.

    The localizer has no access to a state label, entity ID, geometry, or
    simulator metadata.  It merely computes an image-difference bounding box
    from the two method-visible RGB arrays, then includes two tiles of public
    visual context around it.
    """

    import numpy as np
    from PIL import Image

    before_image, after_image = Image.open(before).convert("RGB"), Image.open(after).convert("RGB")
    before_array, after_array = np.asarray(before_image), np.asarray(after_image)
    if before_array.shape != after_array.shape:
        raise ValueError("before/after RGB dimensions differ")
    changed = np.any(before_array != after_array, axis=2)
    ys, xs = np.nonzero(changed)
    if not len(xs):
        return before_image, after_image
    tile = 32
    height, width = changed.shape


    left = max(0, (int(xs.min()) // tile - 2) * tile)
    top = max(0, (int(ys.min()) // tile - 2) * tile)
    right = min(width, ((int(xs.max()) // tile) + 3) * tile)
    bottom = min(height, ((int(ys.max()) // tile) + 3) * tile)
    return before_image.crop((left, top, right, bottom)), after_image.crop((left, top, right, bottom))


class QwenFormalGrounder:
    """Two-image greedy Grounder; it has no evaluator/private dependency."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        max_new_tokens: int = 96,
        use_change_crop: bool = False,
        prompt_version: str = "v1",
        device: str = "cuda",
    ):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration





        torch.backends.cudnn.enabled = False
        self._torch = torch
        self.model_path = Path(model_path).resolve()
        self.max_new_tokens = max_new_tokens
        self.use_change_crop = use_change_crop
        self.prompt_version = prompt_version
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype="auto", device_map={"": device}, local_files_only=True
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
        self.device = device if torch.cuda.is_available() else "cpu"

    def ground(
        self, *, before_image: str | Path, after_image: str | Path, action: str, entity_catalog: Iterable[dict[str, object]]
    ) -> str:
        from qwen_vl_utils import process_vision_info

        before, after = Path(before_image).resolve(), Path(after_image).resolve()
        if not before.is_file() or not after.is_file():
            raise FileNotFoundError(f"missing Grounder image pair: {before}, {after}")
        content: list[dict[str, object]] = [
            {"type": "image", "image": before.as_uri()},
            {"type": "image", "image": after.as_uri()},
        ]
        if self.use_change_crop:
            before_crop, after_crop = _public_change_crops(before, after)
            content.extend(
                [
                    {"type": "image", "image": before_crop},
                    {"type": "image", "image": after_crop},
                ]
            )
            crop_instruction = (
                "Images 3 and 4 are aligned enlarged crops automatically computed from public RGB differences; "
                "use them to inspect the local changed cell, while Images 1 and 2 provide global context."
            )
        else:
            crop_instruction = "Only Images 1 and 2 are provided."
        content.append(
            {
                "type": "text",
                "text": f"{crop_instruction}\n\n{build_grounder_prompt(action=action, entity_catalog=entity_catalog, version=self.prompt_version)}",
            }
        )
        messages = [
            {"role": "system", "content": "Follow the user output contract exactly."},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated, strict=True)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
