import unsloth  # noqa: F401 # isort: skip

import argparse

import datasets
import torch
import yaml
from transformers import AutoProcessor
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

from processors import (
    IconQASample,
    convert_custom_train_to_conversation,
    convert_icon_qa_train_to_conversation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--icon-qa-train-dataset", required=True)
    parser.add_argument("--custom-train-dataset", required=True)
    parser.add_argument("--sft-config", required=True)
    return parser.parse_args()


def build_train_dataset(
    icon_qa_train_dataset_path: str, custom_train_dataset_path: str
) -> datasets.Dataset:
    icon_qa_train_dataset = datasets.Dataset.from_file(icon_qa_train_dataset_path)
    icon_qa_samples = [
        convert_icon_qa_train_to_conversation(IconQASample(**dict(sample)))
        for sample in icon_qa_train_dataset
    ]

    custom_train_dataset = datasets.Dataset.from_file(custom_train_dataset_path)
    custom_samples = [
        convert_custom_train_to_conversation(dict(sample))
        for sample in custom_train_dataset
    ]

    all_samples = []
    for sample in icon_qa_samples + custom_samples:
        images = []
        normalized_prompt = []

        for message in sample["prompt"]:
            content = message["content"]
            if not isinstance(content, list):
                normalized_prompt.append(message)
                continue

            normalized_content = []
            for part in content:
                if part["type"] == "image":
                    images.append(part["image"])
                    normalized_content.append({"type": "image"})
                else:
                    normalized_content.append(part)

            normalized_prompt.append(
                {
                    "role": message["role"],
                    "content": normalized_content,
                }
            )

        for message in sample["completion"]:
            content = message["content"]
            if isinstance(content, list) and any(
                part["type"] == "image" for part in content
            ):
                raise ValueError("Completion messages must not contain images.")

        all_samples.append(
            {
                "images": images,
                "prompt": normalized_prompt,
                "completion": sample["completion"],
            }
        )

    return datasets.Dataset.from_list(all_samples)


class FixedUnslothVisionDataCollator(UnslothVisionDataCollator):
    def _extract_images_for_pc(self, example, p_msgs, c_msgs):
        if "images" in example:
            return self._resize_images_inplace(list(example["images"])), [], None
        return super()._extract_images_for_pc(example, p_msgs, c_msgs)

    def _collate_prompt_completion(self, examples):
        output = super()._collate_prompt_completion(examples)
        mm_token_type_ids = output.get("mm_token_type_ids")
        if (
            mm_token_type_ids is None
            or mm_token_type_ids.shape == output["input_ids"].shape
        ):
            return output

        fixed = torch.zeros_like(output["input_ids"])
        fixed[output["input_ids"] == self.processor.image_token_id] = 1
        fixed[output["input_ids"] == self.processor.video_token_id] = 2
        output["mm_token_type_ids"] = fixed
        return output


def main() -> None:
    args = parse_args()

    base_model, _ = FastVisionModel.from_pretrained(
        "unsloth/Qwen3.5-0.8B-Base",
        load_in_4bit=False,
    )
    processor = AutoProcessor.from_pretrained("unsloth/Qwen3.5-0.8B")

    model = FastVisionModel.get_peft_model(base_model)

    FastVisionModel.for_training(model)

    train_dataset = build_train_dataset(
        args.icon_qa_train_dataset, args.custom_train_dataset
    )

    if len(train_dataset) > 2000:
        raise ValueError(
            "The  training dataset is too large. Please ensure it has at most 2000 samples."
        )

    with open(args.sft_config, "r", encoding="utf-8") as f:
        sft_config = SFTConfig(
            **yaml.safe_load(f),
        )

    if sft_config.max_length is None or sft_config.max_length > 2048:
        raise ValueError("max_length must be set and <= 2048")

    if (
        sft_config.max_steps != -1
        and (
            sft_config.per_device_train_batch_size
            * sft_config.gradient_accumulation_steps
            * sft_config.max_steps
        )
        > 2000
    ):
        raise ValueError("Total training steps must be <= 2000 when using max_steps")

    if sft_config.max_steps == -1 and sft_config.num_train_epochs > 1.0:
        raise ValueError("num_train_epochs must be <= 1.0 when max_steps is not set")

    data_collator = FixedUnslothVisionDataCollator(model, processor)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        data_collator=data_collator,
        train_dataset=train_dataset,
        processing_class=processor,
    )
    trainer.train()

    model.save_pretrained("checkpoints/final")


if __name__ == "__main__":
    main()
