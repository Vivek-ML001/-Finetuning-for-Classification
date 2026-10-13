# GPT-2 Spam Classifier (Built with PyTorch)

Turn a text-generating AI (GPT-2) into a **spam detector** for SMS messages.

> Input: `"You are a winner! You have been selected to receive $1000 cash..."`
> Output: **spam**
>
> Input: `"Hey, are we still on for dinner tonight?"`
> Output: **not spam**

This project follows Chapter 6 of Sebastian Raschka's book *Build a Large Language Model (From Scratch)*.

---

## The Big Idea (in simple words)

GPT-2 was originally trained to do one thing: **guess the next word**.
It can write sentences, but it does not know how to say "spam" or "not spam".

So we do two things:

1. Take a GPT-2 that already understands English (**pretrained model**).
2. Teach it one new skill using examples with answers (**supervised finetuning**).

Think of it like hiring a person who already speaks English well, then training them for one job: sorting spam messages.

---

## Full Pipeline

```
 STAGE 1                STAGE 2                 STAGE 3
 Pretraining            Add a new head          Supervised Finetuning
 (done by OpenAI)       (our change)            (we do this)

 Huge internet text     GPT-2 body              Labeled SMS messages
        |                    |                  ("spam" / "not spam")
        v                    v                         |
 Learn to predict       Replace the last                v
 the next word          layer with a            Update only a few layers
        |               2-class layer                   |
        v                    |                         v
 Pretrained GPT-2  --->  GPT-2 + new head  --->  Spam Classifier
 (124M weights)                                         |
                                                        v
                                              STAGE 4: Test + Use
```

---

## Stage by Stage

### Stage 1: Pretraining (already done by OpenAI)

- GPT-2 was trained on a huge amount of internet text.
- Its only task was: *given some words, predict the next word.*
- Because of this, it learned grammar, meaning, and general language knowledge.
- **We do not repeat this step.** We download the finished weights (GPT-2 small, 124 million parameters).

### Stage 2: Change the Model for Classification

Original GPT-2 ends with a layer that outputs **50,257 values** (one for every word in its vocabulary).

We need only **2 answers** (spam or not spam). So we:

1. **Freeze** all layers (they keep what they already learned).
2. **Replace** the last layer (`out_head`) with a small new layer: `768 -> 2`.
3. **Unfreeze** only:
   - the last transformer block
   - the final layer norm
   - the new output head

```
 Text
  |
  v
 Tokenizer (GPT-2 BPE) ---> token numbers
  |
  v
 Token Embedding + Position Embedding
  |
  v
 [ Transformer Block 1  ]  \
 [ Transformer Block 2  ]   |  FROZEN (not changed)
 [        ...           ]   |
 [ Transformer Block 11 ]  /
 [ Transformer Block 12 ]  <-- TRAINED
  |
  v
 Final LayerNorm            <-- TRAINED
  |
  v
 Output Head (768 -> 2)     <-- NEW + TRAINED
  |
  v
 Take the LAST token's output
  |
  v
 [score_not_spam, score_spam] ---> pick the bigger one
```

**Why the last token?** GPT-2 reads left to right. The last token has "seen" every word before it, so it holds the summary of the whole message.

### Stage 3: Supervised Finetuning

"Supervised" means every example comes with the correct answer (a label).

**Data**

- Dataset: [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- We balanced it: **747 spam + 747 not spam** messages (so the model does not just guess the majority class).
- Split: **70% train / 10% validation / 20% test**
- Short messages are padded with `<|endoftext|>` (token `50256`) so every message has the same length.

**Training setup**

| Setting | Value |
|---|---|
| Base model | GPT-2 small (124M) |
| Loss | Cross-entropy |
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Weight decay | 0.1 |
| Batch size | 8 |
| Epochs | 5 |
| Hardware | Google Colab T4 GPU (about 2 minutes) |

**What happens in each step**

1. Take a batch of 8 messages.
2. Model predicts spam / not spam.
3. Compare with the real label and calculate the loss.
4. Update the trainable layers to reduce the loss.
5. Repeat for all batches, for 5 epochs.

### Stage 4: Evaluate and Use

We check accuracy on **data the model never saw** (the test set).

| Dataset | Accuracy |
|---|---|
| Training | 97.21% |
| Validation | 98.66% |
| Test | 95.33% |

Training curves:

- `loss-plot.pdf`
- `accuracy-plot.pdf`

---

## Try It

```python
import torch
import tiktoken
from previous_chapters import GPTModel

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

# Rebuild the model the same way as in training
model = GPTModel(BASE_CONFIG)
model.out_head = torch.nn.Linear(BASE_CONFIG["emb_dim"], 2)

# Load finetuned weights (download from Hugging Face, link below)
model.load_state_dict(torch.load("review_classifier.pth", map_location=device))
model.to(device)
model.eval()


def classify_review(text, model, tokenizer, device, max_length, pad_token_id=50256):
    input_ids = tokenizer.encode(text)
    input_ids = input_ids[:min(max_length, model.pos_emb.weight.shape[0])]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    return "spam" if torch.argmax(logits, dim=-1).item() == 1 else "not spam"


print(classify_review(
    "You are a winner you have been specially selected to receive $1000 cash.",
    model, tokenizer, device, max_length=120,
))
```

`max_length` should match the value used in training (the longest message in the training set).

---

## Files in This Repo

| File | What it does |
|---|---|
| `finetuning.py` | Main code: data loading, model changes, training, evaluation |
| `gpt_download.py` | Downloads the pretrained GPT-2 weights |
| `previous_chapters.py` | GPT model code (from earlier chapters of the book) |
| `train.csv`, `validation.csv`, `test.csv` | Our data splits |
| `loss-plot.pdf`, `accuracy-plot.pdf` | Training graphs |

The trained weights (`review_classifier.pth`, about 522 MB) are too big for GitHub.
Download them here: **https://huggingface.co/YOUR-HF-USERNAME/gpt2-spam-classifier**

---

## Run It Yourself

```bash
pip install torch tiktoken pandas matplotlib tensorflow tqdm requests
python finetuning.py
```

A GPU (for example a free Colab T4) is recommended.

---

## Key Words (Quick Glossary)

- **Pretraining:** First training on huge general text so the model learns language.
- **Finetuning:** Extra training on a small task-specific dataset.
- **Supervised:** Every training example has a correct label.
- **Token:** A small piece of text (word or part of a word) converted to a number.
- **Freeze:** Stop a layer from changing during training.
- **Logits:** Raw scores the model gives before choosing an answer.
- **Cross-entropy loss:** A number that says how wrong the model's answer is.

---

## Credits

- Book and original code: Sebastian Raschka, [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- Pretrained model: OpenAI GPT-2
- Dataset: UCI SMS Spam Collection

## License

MIT
