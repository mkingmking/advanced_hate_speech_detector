import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model_and_tokenizer(model_dir: str, device: torch.device):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


def infer_examples(model, tokenizer, examples, device):
    enc = tokenizer(
        examples,
        padding="longest",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
    return probs


def evaluate_on_dataset(model, tokenizer, csv_path, device, batch_size=32):
    df = pd.read_csv(csv_path)
    df["tweet"] = df["tweet"].fillna("").astype(str)
    texts = df["tweet"].tolist()
    labels = df["label"].tolist() if "label" in df.columns else None

    all_preds, all_labels = [], []

    model.eval()
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = torch.argmax(logits, dim=1).cpu().tolist()

        all_preds.extend(preds)
        if labels is not None:
            all_labels.extend(labels[start : start + batch_size])

    if labels is not None:
        acc = accuracy_score(all_labels, all_preds)
        f1  = f1_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=["not hate","hate"])
        return acc, f1, report
    else:
        # save predictions only
        out_df = pd.DataFrame({
            "tweet": texts,
            "predicted_label": all_preds
        })
        out_df.to_csv("data/processed/test_predictions.csv", index=False)
        return None, None, None


def main():
    # 1) Device setup
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model_dir = "models/fine_tuned"

    print(f"Loading model from {model_dir} on {device}")
    model, tokenizer = load_model_and_tokenizer(model_dir, device)

    # 2) Quick manual examples
    examples = [
        "mother fucker",
        "trump black president"
    ]
    probs = infer_examples(model, tokenizer, examples, device)
    print("\n=== Manual Examples ===")
    for text, prob in zip(examples, probs):
        print(f"> {text!r}\n  non-hate: {prob[0]:.3f}, hate: {prob[1]:.3f}\n")

    # 3) Evaluation on test dataset
    test_csv = "data/processed/train_raw_pandera_processed.csv"  # adjust path if needed
    print("\n=== Evaluation on Test Set ===")
    try:
        acc, f1, report = evaluate_on_dataset(model, tokenizer, test_csv, device)
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}\n")
        print(report)
    except FileNotFoundError:
        print(f"Test file not found at {test_csv}, skipping evaluation.")

if __name__ == "__main__":
    main()
