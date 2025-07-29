# app/train.py

import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
from pathlib import Path

def load_data(encodings_path, processed_csv_path):
    """
    Loads tokenized encodings (plain tensors) and labels, returns a TensorDataset.
    """
    enc = torch.load(encodings_path)   # now safe, just a dict of tensors
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    df = pd.read_csv(processed_csv_path)
    labels = torch.tensor(df["label"].tolist(), dtype=torch.long)

    return TensorDataset(input_ids, attention_mask, labels)
def get_model(model_name="vinai/bertweet-base", num_labels=2, freeze_base=False):
    """
    Load a pre-trained sequence classification model.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    if freeze_base:
        # e.g. freeze all except the classification head
        for param in model.base_model.parameters():
            param.requires_grad = False
    return model

def compute_class_weights(labels):
    """
    Returns a tensor of shape [num_labels] with inverse-frequency weights.
    """
    counts = torch.bincount(labels)
    weights = 1.0 / counts.float()
    # optional: normalize to sum to num_labels
    weights = weights * (len(counts) / weights.sum())
    return weights

def train(
    encodings_path="data/processed/train_tokenized.pt",
    processed_csv="data/processed/train_raw_pandera_processed.csv",
    model_name="vinai/bertweet-base",
    output_dir="models/fine_tuned",
    epochs=3,
    batch_size=16,
    lr=2e-5,
    max_grad_norm=1.0,
    device=None
):
    # 1) Device
    # Prefer MPS on macOS, then CUDA, then CPU
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print("Using device:", device)

    

    # 2) Data
    dataset = load_data(encodings_path, processed_csv)
    train_sampler = RandomSampler(dataset)
    train_loader  = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

    # 3) Model
    model = get_model(model_name).to(device)

    # 4) Class weights + loss
    all_labels = torch.tensor(pd.read_csv(processed_csv)["label"].tolist())
    class_weights = compute_class_weights(all_labels).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # 5) Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # 6) Training loop
    model.train()
    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        total_loss = 0
        for step, batch in enumerate(train_loader, 1):
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None
            )
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            if step % 100 == 0 or step == len(train_loader):
                avg = total_loss / step
                print(f"  step {step}/{len(train_loader)} — avg loss: {avg:.4f}")

    # 7) Save model & tokenizer
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving model to {out_dir}…")
    model.save_pretrained(out_dir)
    tokenizer = model.config._name_or_path
    print("Done.")

if __name__ == "__main__":
    train()
