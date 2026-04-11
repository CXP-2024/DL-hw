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
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a visual question answering assistant. You will be given a question about an image, along with two candidate answer images labeled choice_0.png and choice_1.png. Select the correct choice and put your answer within \\boxed{}. Answer with only choice_0.png or choice_1.png.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Question image:",
                    },
                    {
                        "type": "image",
                        "image": sample["query_image"],
                    },
                    {
                        "type": "text",
                        "text": f"Question: {sample['question']}",
                    },
                    {
                        "type": "text",
                        "text": "Choice A (choice_0.png):",
                    },
                    {
                        "type": "image",
                        "image": sample["choice_image_0"],
                    },
                    {
                        "type": "text",
                        "text": "Choice B (choice_1.png):",
                    },
                    {
                        "type": "image",
                        "image": sample["choice_image_1"],
                    },
                    {
                        "type": "text",
                        "text": "Which choice is correct? Answer with choice_0.png or choice_1.png inside \\boxed{}.",
                    },
                ],
            },
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

    # Try to find \boxed{...} pattern first
    match = re.search(r"\\boxed\{(.*?)\}", generated_text)
    if match:
        answer = match.group(1).strip()
        # Normalize to expected format
        if "choice_0" in answer or "0" == answer.strip():
            return "choice_0.png"
        if "choice_1" in answer or "1" == answer.strip():
            return "choice_1.png"
        return answer

    # Fallback: look for choice_X.png anywhere in the text
    matches = re.findall(r"choice_[01]\.png", generated_text)
    if matches:
        return matches[-1]

    # Fallback: look for "choice 0" or "choice 1" patterns
    if "choice_0" in generated_text.lower() or "choice a" in generated_text.lower():
        return "choice_0.png"
    if "choice_1" in generated_text.lower() or "choice b" in generated_text.lower():
        return "choice_1.png"

    return ""

    # YOUR CODE END.
