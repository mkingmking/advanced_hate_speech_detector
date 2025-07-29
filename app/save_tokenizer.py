# app/save_tokenizer.py

from transformers import AutoTokenizer

def main():
    MODEL_DIR = "models/fine_tuned"
    # 1) Load the original tokenizer config
    tokenizer = AutoTokenizer.from_pretrained(
        "vinai/bertweet-base",
        use_fast=True,
        normalization=True
    )
    # 2) Save it into your fine-tuned model folder
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Tokenizer saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
