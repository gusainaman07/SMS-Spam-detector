#  SMS Spam Classifier

A machine learning web application that classifies SMS/email messages as **Spam** or **Not Spam (Ham)** in real time. Built with a TF-IDF + Logistic Regression pipeline and deployed as an interactive Streamlit app, the model combines text-based features with hand-engineered spam-indicator signals to improve detection accuracy and explainability.

##  Highlights

- **Real-time classification** with confidence scores for every prediction
- **Explainable results** — surfaces the exact spam-trigger phrases (e.g. "urgent", "claim", "free") found in a message
- **Hybrid feature engineering**: combines TF-IDF text vectors with 30+ curated spam-keyword indicators for stronger signal
- **Class-balanced Logistic Regression** model to handle the natural imbalance between spam and legitimate messages
- **Clean, interactive UI** built with Streamlit, deployable to Heroku out of the box

##  How It Works

1. **Preprocessing** — incoming text is lowercased, stripped of URLs/HTML/punctuation/numbers, tokenized, filtered for stopwords, and stemmed using NLTK's Porter Stemmer.
2. **Feature Extraction** — cleaned text is vectorized with a TF-IDF vectorizer (unigrams + bigrams, top 5,000 features), then concatenated with binary indicator features for known spam-related keywords and phrases.
3. **Classification** — the combined feature vector is passed to a class-weighted Logistic Regression model, which outputs a label (spam/ham) along with a probability score.
4. **Explainability** — the app highlights which spam indicators were detected in the message, giving users insight into *why* it was flagged.

##  Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| ML / NLP | scikit-learn, NLTK (stopwords, Porter Stemmer, tokenizer) |
| Feature Engineering | TF-IDF Vectorization, custom keyword indicators |
| Web Framework | Streamlit |
| Data Handling | Pandas, NumPy |
| Deployment | Heroku-ready (`Procfile`, `setup.sh`) |

##  Model Performance

Evaluated on a held-out test split of the [SMS Spam Collection dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection):

| Metric | Score |
|---|---|
| Accuracy | ~97% |
| Precision (Spam) | ~85% |
| Recall (Spam) | ~92% |
| F1-Score (Spam) | ~88% |

##  Project Structure

```
sms-spam-classifier-main/
├── app.py                     # Streamlit web application
├── train_model.py             # Data preprocessing, training & evaluation script
├── spam.csv                   # SMS Spam Collection dataset
├── sms-spam-detection.ipynb   # Exploratory data analysis & model development notebook
├── enhanced_model.pkl         # Trained Logistic Regression model
├── enhanced_vectorizer.pkl    # Fitted TF-IDF vectorizer
├── requirements.txt           # Python dependencies
├── nltk.txt                   # NLTK corpora required at runtime
├── Procfile / setup.sh        # Heroku deployment configuration
```

##  Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/gusainaman07/SMS-Spam-detector.git
cd SMS-Spam-detector/sms-spam-classifier-main
pip install -r requirements.txt
```

### Train the Model

```bash
python train_model.py
```

This trains a fresh model on `spam.csv` and generates:
- `enhanced_model.pkl` — the trained classifier
- `enhanced_vectorizer.pkl` — the fitted TF-IDF vectorizer

### Run the App

```bash
streamlit run app.py
```

Then open the local URL Streamlit provides, enter a message, and click **Check for Spam** to see the prediction, confidence score, and detected spam indicators.

##  Future Improvements

- Experiment with deep learning approaches (LSTM/BERT) for improved accuracy on nuanced messages
- Expand the spam-indicator lexicon using automated feature importance analysis
- Add multilingual support for non-English SMS content
- Containerize the app with Docker for simpler deployment

##  License

This project is open source and available for personal and educational use.
