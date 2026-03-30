import unsloth  # noqa: F401 # isort: skip

import argparse

import datasets
import torch
import tqdm
from transformers import AutoProcessor
from unsloth import FastVisionModel

from processors import (
    Conversation,
    IconQASample,
    convert_icon_qa_test_to_conversation,
    extract_answer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", default="unsloth/Qwen3.5-0.8B-Base")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    model, _ = FastVisionModel.from_pretrained(args.checkpoint, load_in_4bit=False)
    processor = AutoProcessor.from_pretrained("unsloth/Qwen3.5-0.8B")

    FastVisionModel.for_inference(model)

    dataset = datasets.Dataset.from_file(args.dataset)

    num_correct = 0
    for sample in tqdm.tqdm(dataset):
        conversation: Conversation = convert_icon_qa_test_to_conversation(
            IconQASample(
                **{k: v for k, v in dict(sample).items() if k != "answer"},
                answer=None,
            ),
        )["messages"]

        inputs = processor.apply_chat_template(
            conversation=conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.to(model.device)

        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_length=2048,
        )
        generated_text = processor.post_process_image_text_to_text(
            outputs[:, inputs["input_ids"].shape[1] :]
        )[0]

        prediction = extract_answer(generated_text)

        if prediction == dict(sample)["answer"]:
            num_correct += 1

    accuracy = num_correct / len(dataset)

    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
