# Task 2: Supervised Fine-Tuning of GPT-2 on Alpaca

## Overview

This project fine-tunes a pretrained GPT-2 model on the Stanford Alpaca dataset using **Supervised Fine-Tuning (SFT)**. The goal is to turn GPT-2 into an instruction-following assistant that can generate helpful responses for user prompts.

Instead of training a language model from scratch, this project uses **transfer learning** by starting from a pretrained model and adapting it to a downstream conversational task.

---

## Features

* Loads the Alpaca dataset from Hugging Face
* Formats data using the required instruction prompt template
* Splits dataset into train / validation / test sets
* Loads pretrained GPT-2 (`distilgpt2`) model and tokenizer
* Adds a real `[PAD]` token for padding support
* Tokenizes examples for causal language modeling
* Fine-tunes the model using PyTorch
* Tracks training and validation loss
* Generates assistant-style responses after training
* Plots loss curve after training

---


## Training and validation loss:
<img width="627" height="658" alt="Screenshot 2026-04-11 134208" src="https://github.com/user-attachments/assets/c9becfd0-2781-43e9-b132-3e32d99323b8" />

## One sample of generated response and plotting of Training and validation loss:
<img width="1675" height="712" alt="Screenshot 2026-04-11 134226" src="https://github.com/user-attachments/assets/6aa83060-534b-4dbd-95cb-f9c8771d7c5d" />

## Sample of some generated response:
<img width="1772" height="269" alt="Screenshot 2026-04-11 134532" src="https://github.com/user-attachments/assets/51d858fb-4501-4de8-a99c-959051d90bb4" /> 
<img width="1733" height="295" alt="Screenshot 2026-04-11 134948" src="https://github.com/user-attachments/assets/b2095fcb-fb44-4e94-b808-98f541f57fb7" />
<img width="1796" height="430" alt="Screenshot 2026-04-11 134832" src="https://github.com/user-attachments/assets/2ab8f855-0111-4ae1-a5a4-54370cdb50e8" /> 
## How to Run

### Install Dependencies

```bash id="4xxe5u"
pip install torch transformers datasets matplotlib
```

### Run Training

```bash id="2jpk2x"
python train.py
```

This will:

* Load and preprocess dataset
* Fine-tune GPT-2
* Show training progress
* Generate sample output
* Save loss plot

### Run Inference

```bash id="6r9j0l"
python inference.py
```

This will let you test the fine-tuned model on new instructions.
