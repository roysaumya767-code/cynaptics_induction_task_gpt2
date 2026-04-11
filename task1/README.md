# GPT-2 Style Character-Level Language Model (Tiny Shakespeare)

## 📌 Overview

This project implements a **decoder-only Transformer (GPT-2 style)** language model from scratch using **PyTorch**.
The model is trained on the **Tiny Shakespeare dataset** to perform **next-token prediction**, effectively acting as a **glorified autocomplete system**.

Given a prompt, the model generates Shakespeare-like text by sampling from learned probability distributions.

---

## 🚀 Features

* Custom **Byte Pair Encoding (BPE)** tokenizer
* Decoder-only Transformer architecture
* Masked Multi-Head Self-Attention
* Feedforward Neural Network (MLP block)
* Residual connections + Layer Normalization
* Text generation via probabilistic sampling

---

## 📂 Dataset

We use the **Tiny Shakespeare dataset** (~40K lines of text).

Dataset is automatically downloaded from:
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

---

## ⚙️ Model Architecture

### Embeddings

* Token Embedding: `vocab_size × n_embd`
* Positional Embedding: `block_size × n_embd`

### Transformer Block (repeated `n_layer` times)

Each block consists of:

1. **Masked Multi-Head Self-Attention**
2. **Feed Forward Network**
3. **Residual Connections**
4. **Layer Normalization**

### Output Head

* Linear layer projecting to `vocab_size`
* Softmax applied during loss computation

---

## 🔢 Hyperparameters

| Parameter      | Value |
| -------------- | ----- |
| Batch Size     | 64    |
| Block Size     | 256   |
| Embedding Size | 384   |
| Heads          | 6     |
| Layers         | 6     |
| Dropout        | 0.2   |
| Learning Rate  | 3e-4  |
| Max Iterations | 5000  |

---

## 🧠 Tokenization (BPE)

* Starts from raw UTF-8 byte encoding (256 tokens)
* Performs merges based on frequency
* Final vocabulary size: **276 tokens**

---

## 🏋️ Training

The model is trained using:

* **Cross-Entropy Loss**
* **AdamW Optimizer**

Loss is evaluated periodically on:

* Training split (90%)
* Validation split (10%)

### Run Training

```bash
python train.py
```

---

## ✨ Text Generation (Autocomplete)

After training, the model generates text using:

* Softmax probabilities
* Multinomial sampling

## 🧩 The Training and validation loss are shown here:
<img width="532" height="279" alt="Screenshot 2026-04-09 195501" src="https://github.com/user-attachments/assets/23013f19-21be-4d18-bda5-58949d721ba0" />

## 🧩 The Text generation results are:
<img width="595" height="840" alt="Screenshot 2026-04-09 195533" src="https://github.com/user-attachments/assets/60936348-f956-43c1-9d27-a2b51b6c1aca" />

### Masked Self-Attention

Ensures the model only attends to **past tokens**, preventing information leakage from the future.

### Multi-Head Attention

Allows the model to learn **different relationships in parallel**.

### Residual Connections

Help stabilize deep networks and improve gradient flow.

### BPE Tokenization

Balances between character-level and word-level modeling.

---

## 📉 Training Behavior

* Training loss decreases over iterations
* Validation loss stabilizes after sufficient training
* Generated text becomes more structured and language-like

---

## 🛠️ How to Run

### 1. Install Dependencies

```bash
pip install torch requests
```

### 2. Run Training

```bash
python train.py
```

### 3. Generate Text

Generation runs automatically at the end of training.

---

## 📚 References

* "Attention Is All You Need" (Vaswani et al.)
* Andrej Karpathy - *Let's build GPT from scratch*
* Tiny Shakespeare Dataset

---

## ✅ Conclusion

This project demonstrates how a Transformer-based language model can be built from scratch and trained to generate coherent text. While small in scale, it captures the essential ideas behind modern large language models like GPT-2.
