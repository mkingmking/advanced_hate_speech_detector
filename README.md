# Advanced Hate Speech Detector

This repository contains a project for training and serving a hate speech classification model using the [BERTweet](https://github.com/VinAIResearch/BERTweet) architecture. The code includes utilities for ingesting and cleaning tweet data, tokenising text, training a transformer model and deploying a simple Flask web interface for predictions.

## Features

- Data ingestion with optional schema enforcement via [Pandera](https://pandera.readthedocs.io/)
- Pre-processing helpers for tweets (normalisation, stopword removal, etc.)
- Tokenisation using BERTweet or BERT-base
- Training script built on `transformers` and `torch`
- Basic evaluation and prediction utilities
- Flask app with a dashboard and prediction endpoint



## Setup

1. Create a Python environment (Python 3.10+ recommended).
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```


## Usage

### Data Processing

Use `app/file_ingestion.py` to validate and clean raw CSV files:

```bash
python3 -m app.file_ingestion 
```

### Tokenisation

```bash
python3 -m app.tokenization
```

### Training

```bash
python3 -m app.train
```

###Running the API & Dashboard

Start the Flask app:

```bash
python3 -m app.main
```



-Dashboard: View real-time EDA visualisations at http://localhost:5000/dashboard:

-Prediction UI: Classify custom tweets at http://localhost:5000/predict-ui:

-Prediction Endpoint: Send JSON to /predict:
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text":"insert your twit here"}'
```


## Contributing

Feel free to open issues or pull requests. All contributions should pass basic linting (`python -m py_compile app/*.py`) before submission.

## License

This project is released under the MIT License. See `LICENSE` for details.
