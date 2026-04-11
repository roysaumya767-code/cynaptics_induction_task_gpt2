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
