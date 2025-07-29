from flask import Flask, render_template, request, url_for
import pandas as pd
import plotly.express as px

app = Flask(__name__)

# Load once at startup
DF = pd.read_csv("data/processed/train_raw_pandera_processed.csv")
DF["length"] = DF["tweet"].str.len()

@app.route("/dashboard")
def dashboard():
    # 1) Label distribution chart
    label_counts = DF["label"].value_counts().sort_index()
    fig1 = px.bar(
        x=["Not Hate","Hate"], 
        y=label_counts.values, 
        labels={"x":"Label","y":"Count"}
    )
    label_dist_html = fig1.to_html(full_html=False)

    # 2) Length histogram
    fig2 = px.histogram(
        DF, x="length", nbins=50, 
        labels={"length":"Tweet Length (chars)"}
    )
    length_hist_html = fig2.to_html(full_html=False)

    # 3) Sample tweets filtered by length
    try:
        min_len = int(request.args.get("min_len", 0))
        max_len = int(request.args.get("max_len", DF["length"].max()))
    except ValueError:
        min_len, max_len = 0, int(DF["length"].max())

    subset = DF[(DF["length"]>=min_len)&(DF["length"]<=max_len)]
    samples = subset[["label","tweet"]].sample(5).to_dict("records")

    return render_template(
        "dashboard.html",
        label_dist=label_dist_html,
        length_hist=length_hist_html,
        min_len=min_len,
        max_len=max_len,
        samples=samples
    )

if __name__ == "__main__":
    app.run(debug=True)
