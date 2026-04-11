# Deep Learning Coding Project 4: Vision-Language Model Fine-Tuning Report

## Cover Information

- **Name:** Pan Changxun
- **Student ID:** 2024011323

## Generative AI Usage Disclosure

Claude Code (Anthropic) was used to assist with code implementation, hyperparameter tuning, data curation, and report writing. The tool helped with designing the prompt format in `processors.py`, downloading and sampling additional training data from the full IconQA dataset, running hyperparameter search experiments, and structuring the report.

## Custom Data Curation

1,000 additional training samples were curated from the full IconQA dataset (HuggingFace: `lmms-lab/ICON-QA`) and saved as `custom.arrow`.

**Schema**: Same as the provided IconQA data — `question_id`, `question`, `choices`, `answer`, `query_image`, `choice_image_0`, `choice_image_1`, `ques_type`, `label`, `grade`, `skills`.

**Data source**: The full IconQA `choose_img` validation and test splits (~23,000 samples). Samples already present in the provided training set and validation set were excluded by `question_id`. Only 2-choice samples (answer is `choice_0.png` or `choice_1.png`) were kept.

**Sampling strategy**: Strategic sampling to address under-represented categories in the provided 1,000 training samples:
- **Phase 1 (rare skill boost)**: Prioritized skills with fewer samples — `algebra` (80), `time` (80), `commonsense` (80), `fraction` (80), `pattern` (80). These skills had only 15-92 samples in the original data.
- **Phase 2 (proportional fill)**: Remaining 600 slots filled by random sampling from the candidate pool.

**Final custom data distribution**:
- Skills: geometry (355), counting (208), fraction (149), commonsense (137), pattern (128), comparing (123), time (114), spatial (114), scene (95), algebra (104), probability (43)
- Grades: kindergarten (340), grade2 (296), grade1 (190), prek (174)
- Answers: choice_0.png (517), choice_1.png (483) — well balanced

**Total training samples**: 1,000 (provided) + 1,000 (custom) = 2,000

## Prompt and Answer Formatting

### Prompt Design

The prompt was structured as a multi-turn conversation with three key design choices:

1. **System message**: A dedicated system message defines the assistant's role as a "visual question answering assistant", specifying the expected task format and answer structure. This provides consistent task framing across all samples.

2. **Structured user message**: The user message is organized with explicit labels for each visual and textual element:
   - "Question image:" followed by the query image
   - "Question: {question}" for the text question
   - "Choice A (choice_0.png):" and "Choice B (choice_1.png):" as labeled headers before each choice image
   - A clear instruction: "Which choice is correct? Answer with choice_0.png or choice_1.png inside \boxed{}."

3. **Explicit choice labels**: Each choice image is labeled with both a letter (A/B) and its filename (choice_0.png/choice_1.png). This creates a clear mapping between the visual options and the expected answer format.

### Answer Format

The model is trained to output answers in `\boxed{choice_X.png}` format, which is concise and easy to parse.

### Answer Extraction

The `extract_answer` function uses a multi-level fallback strategy:
1. **Primary**: Regex match for `\boxed{...}` pattern, with normalization to `choice_0.png` or `choice_1.png`
2. **Fallback 1**: Search for `choice_X.png` patterns anywhere in the generated text
3. **Fallback 2**: Detect "choice_0", "choice_1", "choice a", or "choice b" mentions

This robust extraction handles cases where the model may not perfectly follow the boxed format.

## Training Configuration

| Hyperparameter | Value |
|---|---|
| `max_length` | 2048 |
| `num_train_epochs` | 1.0 |
| `max_steps` | -1 |
| `per_device_train_batch_size` | 2 |
| `gradient_accumulation_steps` | 4 |
| `learning_rate` | 1e-4 |
| `lr_scheduler_type` | cosine |
| `warmup_ratio` | 0.1 |
| `weight_decay` | 0.01 |
| `bf16` | true |

**Effective batch size**: 2 x 4 x 1 GPU = 8. **Total steps**: 250 (2,000 samples / 8 batch).

**Learning rate selection**: The learning rate of 1e-4 was selected through a hyperparameter search over {5e-5, 1e-4, 1.5e-4, 2e-4, 3e-4}. Results showed that 1e-4 was the sweet spot:

| Learning Rate | Validation Accuracy | Notes |
|---|---|---|
| 5e-5 | 0.845 | Under-fitting, loss converges too slowly |
| **1e-4** | **0.88 - 0.895** | Best accuracy and fast evaluation |
| 1.5e-4 | — | Evaluation very slow (poor EOS learning) |
| 2e-4 | 0.860 | Slightly worse, EOS learning unstable |
| 3e-4 | — | Evaluation extremely slow, stopped early |

Higher learning rates (>=1.5e-4) caused the model to poorly learn the EOS token, resulting in very long generation during inference.

**Other design choices**:
- **Cosine scheduler with warmup**: The warmup ratio of 0.1 prevents early instability, while cosine decay ensures smooth convergence.
- **bf16 precision**: Reduces memory footprint and speeds up training without significant precision loss.
- **1 epoch**: Sufficient for 2,000 samples; the training loss converges from 0.39 to 0.02 within a single epoch.

## Results

| Model | Validation Accuracy |
|---|---|
| Base model (zero-shot) | 0.300 |
| Trained (1000 samples, lr=2e-4) | 0.875 |
| **Trained (2000 samples, lr=1e-4)** | **0.88 - 0.895** |

### Training Loss Curve (Final Configuration)

| Step | Loss | Learning Rate |
|---|---|---|
| 10 | 0.3857 | 3.6e-5 |
| 20 | 0.0476 | 7.6e-5 |
| 50 | 0.0393 | 9.7e-5 |
| 100 | 0.0280 | 7.6e-5 |
| 150 | 0.0155 | 4.9e-5 |
| 200 | 0.0282 | 1.2e-5 |
| 250 | 0.0203 | ~0 |

### Discussion

**What helped performance:**

1. **Custom training data (+1,000 samples)**: Adding strategically sampled data from the full IconQA dataset increased training diversity. Combined with a tuned learning rate, this improved accuracy from 0.875 to ~0.89.

2. **Learning rate tuning**: Reducing the learning rate from 2e-4 to 1e-4 was critical when training with 2,000 samples. Higher learning rates caused the model to not properly learn the EOS token, leading to long generation at inference time and slightly worse accuracy.

3. **Structured prompt with system message**: The system message and labeled choice images provide clear task framing that the model can leverage during both training and inference.

4. **Robust answer extraction**: The multi-level fallback in `extract_answer` ensures answers are captured even when the model's output format is slightly imperfect.

5. **Strategic data sampling**: Boosting under-represented skills (algebra, time, commonsense, etc.) improved the model's coverage across question types.

**What did not help or hurt:**

- **Higher learning rates (>=1.5e-4)**: Caused poor EOS token learning, making inference extremely slow (hours instead of minutes for 200 samples).
- **Lower learning rate (5e-5)**: Under-fitted with only 1 epoch of training, resulting in lower accuracy (0.845).

**Observations:**

- The base model achieves 30% accuracy in zero-shot, below random chance (50%) for binary choice, suggesting it struggles with the multimodal QA format without fine-tuning.
- After 1 epoch of SFT with 2,000 samples, accuracy reaches ~0.89, a ~59 percentage point improvement.
- Training loss converges rapidly (within the first 20 steps) and remains low, suggesting the task is well within the model's capacity with proper LoRA fine-tuning (1.52% of parameters trained).
- There is ~1-2% accuracy variance between runs due to training randomness (data shuffle, LoRA initialization).
