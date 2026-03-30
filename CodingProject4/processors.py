import re
from typing import Any, TypedDict

from PIL.Image import Image

type Conversation = list[dict[str, Any]]


class ConversationalLanguageModeling(TypedDict):
    messages: Conversation


class ConversationalPromptCompletion(TypedDict):
    prompt: Conversation
    completion: Conversation


class IconQASample(TypedDict):
    question: str
    choices: str
    answer: str | None
    query_image: Image
    choice_image_0: Image
    choice_image_1: Image


def convert_custom_train_to_conversation(
    sample: dict[str, Any],
) -> ConversationalPromptCompletion:
    """Builds one SFT conversation from a custom training sample.

    Args:
        sample: A sample in the custom training dataset. The schema of this
            dataset is student-defined.

    Returns:
        A conversation for training. You are responsible for converting your
        custom sample format into this prompt-completion structure.
    """

    # YOUR CODE BEGIN.

    return convert_icon_qa_train_to_conversation(IconQASample(**sample))

    # YOUR CODE END.


def convert_icon_qa_test_to_conversation(
    sample: IconQASample,
) -> ConversationalLanguageModeling:
    """Builds one eval conversation from an IconQA sample.

    Args:
        sample: A IconQA sample, whose ``answer`` field is always ``None``.

    Returns:
        A conversation for testing.
    """

    # YOUR CODE BEGIN.

    return ConversationalLanguageModeling(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": sample["query_image"],
                    },
                    {
                        "type": "text",
                        "text": sample["question"],
                    },
                    {
                        "type": "image",
                        "image": sample["choice_image_0"],
                    },
                    {
                        "type": "image",
                        "image": sample["choice_image_1"],
                    },
                    {
                        "type": "text",
                        "text": f"Choices: {sample['choices']}",
                    },
                    {
                        "type": "text",
                        "text": r"Put your answer within \boxed{}",
                    },
                ],
            }
        ]
    )

    # YOUR CODE END.


def convert_icon_qa_train_to_conversation(
    sample: IconQASample,
) -> ConversationalPromptCompletion:
    """Builds one SFT conversation from an IconQA training sample.

    Args:
        sample: A IconQA sample.

    Returns:
        A conversation for training, where the prompt is the same as the test conversation
    """

    # YOUR CODE BEGIN.

    return ConversationalPromptCompletion(
        prompt=convert_icon_qa_test_to_conversation(sample)["messages"],
        completion=[
            {
                "role": "assistant",
                "content": f"\\boxed{{{sample['answer']}}}",
            }
        ],
    )

    # YOUR CODE END.


def extract_answer(generated_text: str) -> str:
    """Extracts the final answer token from model output.

    Args:
        generated_text: Raw generated text.

    Returns:
        The parsed answer.
    """

    # YOUR CODE BEGIN.

    match = re.search(r"\\boxed\{(.*?)\}", generated_text)

    return match.group(1).strip() if match else ""

    # YOUR CODE END.
