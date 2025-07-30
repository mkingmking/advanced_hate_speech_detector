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
python app/file_ingestion.py
```

### Tokenisation

```bash
python app/tokenization.py
```

### Training

```bash
python app/train.py
```

### Running the API

```bash
python app/main.py
```

The Flask application exposes a small dashboard at `/dashboard` and a JSON prediction endpoint at `/predict`.

## Contributing

Feel free to open issues or pull requests. All contributions should pass basic linting (`python -m py_compile app/*.py`) before submission.

## License

This project is released under the MIT License. See `LICENSE` for details.
