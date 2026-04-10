# GPT-2 Fine-Tuning on Alpaca Dataset

## 📌 Overview

This project fine-tunes a pretrained **GPT-2 (distilgpt2)** model on the **Alpaca dataset** to generate instruction-following responses.

---

## 🚀 Features

* Uses pretrained **distilgpt2** (lightweight GPT-2)
* Fine-tuned on **Alpaca instruction dataset**
* Training + validation loss tracking

---

## ⚙️ Hyperparameters

| Parameter           | Value      |
| ------------------- | ---------- |
| Model               | distilgpt2 |
| Batch Size          | 4          |
| Learning Rate       | 5e-5       |
| Epochs              | 1          |
| Max Sequence Length | 128        |
| Optimizer           | AdamW      |
| Eval Frequency      | 1000 steps |
| Eval Batches        | 5          |

---

