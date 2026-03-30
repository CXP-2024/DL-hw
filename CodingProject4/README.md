# Deep Learning Coding Project 4: Vision-Language Model Fine-Tuning

## 1. Overview

In this coding project, you will improve a vision-language multiple-choice QA system based on `unsloth/Qwen3.5-0.8B-Base`.

You will work with the provided IconQA data, optionally add your own custom training data, design the prompt and answer format, tune supervised fine-tuning (SFT) hyperparameters, and document your approach in a short report.

## 2. Environment Setup

This project uses `uv` for environment management.

1. Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If you encounter network issues, you may use the mirror:

```bash
curl -LsSf https://gitee.com/wangnov/uv-custom/releases/download/latest/uv-installer-custom.sh | sh
```

2. Install dependencies:

```bash
uv sync
```

3. Download the provided datasets:

```bash
bash download_data.sh
```

This creates:

- `data/icon-qa-train.arrow`
- `data/icon-qa-val.arrow`

## 3. Dataset

| Property | Value |
| --- | --- |
| Provided training examples | 1,000 |
| Additional custom training examples | Up to 1,000 |
| Validation examples | 200 |
| Test examples for grading | Up to 1,000 |
| Maximum sequence length | 2048 |

## 4. Codebase

The main project files are:

```text
deep-learning-coding-project-4/
├── custom.arrow              # [CREATE THIS] Your custom training dataset (can be empty)
├── data/
│   ├── icon-qa-train.arrow   # [READ-ONLY] Provided training set
│   └── icon-qa-val.arrow     # [READ-ONLY] Provided validation set
├── download_data.sh          # [READ-ONLY] Script to download datasets
├── evaluate.py               # [READ-ONLY] Evaluation script
├── processors.py             # [EDIT THIS] Sample processing functions
├── report.md / report.pdf    # [CREATE THIS] Your report
├── sft_config.yaml           # [EDIT THIS] SFT training configuration
├── train.py                  # [READ-ONLY] Training script
├── pyproject.toml            # [READ-ONLY] Project configuration for uv
└── uv.lock                   # [READ-ONLY] Locked dependencies
```

Files you are expected to work on:

- `processors.py`
- `sft_config.yaml`
- `custom.arrow`
- `report.md` or `report.pdf`

You may create temporary local outputs while experimenting, but they are not part of the submission.

## 5. Tasks

Your goal is to improve model performance while staying within the assignment constraints below.

### Task 1: Prompt and Answer Formatting

Customize `processors.py` to define how IconQA samples are converted into multimodal conversations and how answers are extracted.

Relevant functions:

- `convert_icon_qa_test_to_conversation`
- `convert_icon_qa_train_to_conversation`
- `convert_custom_train_to_conversation`
- `extract_answer`

### Task 2: Custom Training Data

You may add custom training data in Arrow format as `custom.arrow`.

Requirements:

- The dataset must contain at most 1,000 samples; if you do not want to add extra training samples, you must still provide a valid empty Arrow dataset file.
- The file must be named `custom.arrow`.
- The schema of `custom.arrow` is student-defined. You should implement `convert_custom_train_to_conversation` in `processors.py` so that each sample in your custom dataset is converted into a valid training conversation.
- If you choose to reuse the same schema as the provided IconQA data, the starter implementation in `processors.py` is a reasonable baseline.

In other words, the training script only requires that `custom.arrow` can be loaded and that each sample can be converted by your `convert_custom_train_to_conversation` implementation.

### Task 3: SFT Configuration

Customize `sft_config.yaml` to choose an effective training setup.

Requirements:

- `max_length` must be set and must be at most `2048`.
- If `max_steps != -1`, the effective training budget `per_device_train_batch_size * gradient_accumulation_steps * max_steps` must be at most `2000`.
- If `max_steps == -1`, `num_train_epochs` must be at most `1.0`.

### Task 4: Report (`report.md` or `report.pdf`)

Create `report.md` or `report.pdf` in the repository root with the following sections:

- Cover Information: Your name and student ID.
- Generative AI Usage Disclosure: State `None` if you did not use AI. Otherwise, describe which tool(s) you used and how.
- Custom Data Curation: Describe how you constructed `custom.arrow`, including its schema, data source, filtering or cleaning steps, and the number of samples you added.
- Prompt and Answer Formatting: Describe your prompt design, conversation formatting, answer format, and answer extraction strategy in `processors.py`.
- Training Configuration: Describe your SFT setup and training choices, including key hyperparameters such as batch size, gradient accumulation steps, learning rate, optimizer, scheduler, epochs or max steps, and any other important settings in `sft_config.yaml`.
- Results: Include at least your validation accuracy for the base model and your trained checkpoint, along with a brief discussion of what helped or hurt performance.

## 6. Training and Evaluation

If you have trouble downloading models from Hugging Face during training or evaluation, you may use the mirror endpoint:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Evaluate the base model

```bash
uv run python evaluate.py --dataset data/icon-qa-val.arrow
```

This gives you a zero-shot baseline on the validation set.

### Train a LoRA adapter

Before training, make sure you have prepared:

- the provided IconQA training dataset
- a `custom.arrow` file
- an SFT config YAML file

Example:

```bash
uv run python train.py \
  --icon-qa-train-dataset data/icon-qa-train.arrow \
  --custom-train-dataset custom.arrow \
  --sft-config sft_config.yaml
```

After training, the LoRA adapter will be saved to:

- `checkpoints/final`

### Evaluate a trained checkpoint

```bash
uv run python evaluate.py \
  --dataset data/icon-qa-val.arrow \
  --checkpoint checkpoints/final
```

Use this validation result in your report.

## 7. Submission

Your submission must be a `.zip` file containing exactly these four items:

- `processors.py`
- `sft_config.yaml`
- `custom.arrow`
- exactly one report file: `report.md` or `report.pdf`

Do not submit any additional files.

Your report file must be named exactly `report.md` or `report.pdf`.

## 8. Grading

Your project will be evaluated holistically based on the following criteria. The weights indicate relative emphasis, but scores are assigned through an overall review rather than a fixed numerical formula.

| Criteria | Weight | Description |
| --- | --- | --- |
| Prompting | 30% | Evaluated based on the quality of your zero-shot prompting results, with stronger performance generally leading to a stronger overall assessment. Zero-shot accuracy must be at least 0.25. |
| Training | 50% | Evaluated based on how effectively training improves performance relative to a reasonable baseline. Accuracy for the trained model must be at least 0.7. |
| Report | 20% | Evaluated based on the completeness and clarity of the report. |
| Bonus | +1 point | The submission with the **highest accuracy** on our private test set receives **1 bonus point** toward the **final course grade**. |
| Bonus | +0.5 points | The submissions with the **second and third highest accuracy** on our private test set each receive **0.5 bonus points** toward the **final course grade**. |

**Grading Environment:**

Your submission will be executed on the grading platform which at least meets the following specifications.

| Resource | Specification |
|---|---|
| GPU VRAM | 32 GB |
| System RAM | 64 GB |

## 9. Academic Integrity

**Plagiarism in any form will result in an F for the course.**

All submitted code must be your own work. You may discuss high-level ideas with classmates, but sharing or copying code is strictly prohibited.

You must disclose any use of generative AI tools in the appropriate section of your report. If you did not use AI, state `None`. If you did, describe which tool(s) you used and how.
