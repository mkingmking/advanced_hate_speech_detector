import os
import torch
import pandas as pd
from flask import Flask, render_template, request, jsonify, url_for, redirect
import plotly.express as px
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --------------------
# Model & Tokenizer Setup
# --------------------
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model_dir = "models/fine_tuned"

# Load model and tokenizer once
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model.eval()

# --------------------
# Flask App Initialization
# --------------------
app = Flask(__name__)

# --------------------
# Load Data for Dashboard
# --------------------
DF = pd.read_csv("data/processed/train_raw_pandera_processed.csv")
# Use cleaned tweets for length filtering
DF["tweet"] = DF["tweet"].fillna("").astype(str)
DF["length"] = DF["tweet"].str.len()

# --------------------
# Routes
# --------------------
@app.route("/")
def index():
    return redirect(url_for("dashboard"))


@app.route("/health", methods=["GET"])
def health():
    """
    Liveness/readiness check.
    """
    return {"status": "ok"}, 200


@app.route("/dashboard")
def dashboard():
    # Label distribution bar chart
    label_counts = DF["label"].value_counts().sort_index()
    fig1 = px.bar(x=["Not Hate","Hate"], y=label_counts.values,
                  labels={"x":"Label","y":"Count"})
    label_dist_html = fig1.to_html(full_html=False)

    # Tweet length histogram
    fig2 = px.histogram(DF, x="length", nbins=50,
                        labels={"length":"Tweet Length (chars)"})
    length_hist_html = fig2.to_html(full_html=False)

    # Interactive sampling
    try:
        min_len = int(request.args.get("min_len", 0))
        max_len = int(request.args.get("max_len", DF["length"].max()))
    except ValueError:
        min_len, max_len = 0, int(DF["length"].max())
    subset = DF[(DF["length"] >= min_len) & (DF["length"] <= max_len)]
    samples = subset[["label","tweet"]].sample(5).to_dict("records")

    return render_template(
        "dashboard.html",
        label_dist=label_dist_html,
        length_hist=length_hist_html,
        min_len=min_len,
        max_len=max_len,
        samples=samples
    )

@app.route("/predict-ui", methods=["GET"])
def predict_ui():
    # Just render a form; no data needed server-side
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST JSON:
    {
      "text": "single tweet",
      "texts": ["tweet1", "tweet2"]  # optional
    }

    Returns a JSON list of predictions with probabilities.
    """
    data = request.get_json(force=True)
    examples = data.get("texts") or [data.get("text", "")]

    # Tokenize and infer
    enc = tokenizer(
        examples,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().tolist()

    # Build response
    results = []
    for text, p in zip(examples, probs):
        results.append({
            "text": text,
            "non_hate": p[0],
            "hate": p[1],
            "prediction": int(p[1] > 0.5)
        })
    return jsonify(results)

# --------------------
# Entry Point
# --------------------
if __name__ == "__main__":
    # Bind to all interfaces, port configurable via env
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)
