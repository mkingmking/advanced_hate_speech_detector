import os
from pathlib import Path
import torch
import pandas as pd
from flask import Flask, render_template, request, jsonify, url_for, redirect
import plotly.express as px
from transformer import load_model_and_tokenizer, infer_examples
from app.utils import select_device

# --------------------
# Model & Tokenizer Setup
# --------------------
MODEL_DIR = Path("models/fine_tuned")
DATA_PATH = Path("data/processed/train_raw_pandera_processed.csv")

# --------------------
# Flask App Initialization
# --------------------

def load_dashboard_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["tweet"] = df["tweet"].fillna("").astype(str)
    df["length"] = df["tweet"].str.len()
    return df


def create_app(model_dir: Path = MODEL_DIR, csv_path: Path = DATA_PATH) -> Flask:
    device = torch.device(select_device())
    model, tokenizer = load_model_and_tokenizer(str(model_dir), device)
    df = load_dashboard_data(csv_path)

    app = Flask(__name__)

    @app.route("/")
    def index():
        return redirect(url_for("dashboard"))

    @app.route("/health", methods=["GET"])
    def health():
        return {"status": "ok"}, 200

    @app.route("/dashboard")
    def dashboard():
        label_counts = df["label"].value_counts().sort_index()
        fig1 = px.bar(x=["Not Hate", "Hate"], y=label_counts.values,
                      labels={"x": "Label", "y": "Count"})
        label_dist_html = fig1.to_html(full_html=False)

        fig2 = px.histogram(df, x="length", nbins=50,
                            labels={"length": "Tweet Length (chars)"})
        length_hist_html = fig2.to_html(full_html=False)

        try:
            min_len = int(request.args.get("min_len", 0))
            max_len = int(request.args.get("max_len", df["length"].max()))
        except ValueError:
            min_len, max_len = 0, int(df["length"].max())
        subset = df[(df["length"] >= min_len) & (df["length"] <= max_len)]
        samples = subset[["label", "tweet"]].sample(5).to_dict("records")

        return render_template(
            "dashboard.html",
            label_dist=label_dist_html,
            length_hist=length_hist_html,
            min_len=min_len,
            max_len=max_len,
            samples=samples,
        )

    @app.route("/predict-ui", methods=["GET"])
    def predict_ui():
        return render_template("predict.html")

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json(force=True)
        examples = data.get("texts") or [data.get("text", "")]
        probs = infer_examples(model, tokenizer, examples, device)

        results = []
        for text, p in zip(examples, probs):
            results.append({
                "text": text,
                "non_hate": p[0],
                "hate": p[1],
                "prediction": int(p[1] > 0.5),
            })
        return jsonify(results)

    return app


app = create_app()

# --------------------
# Entry Point
# --------------------
if __name__ == "__main__":
    # Bind to all interfaces, port configurable via env
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)
