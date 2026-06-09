# 📧 Spam Email Detector

A Machine Learning model that classifies emails as **Spam** or **Not Spam** using **Support Vector Machine (LinearSVC)** with **TF-IDF** text vectorization.

---

## 📌 Description

This project trains a text classification model on a labeled email dataset to detect spam messages. It uses TF-IDF to convert raw email text into numerical features, then feeds them into a LinearSVC classifier. The trained model is saved to disk and can be reused for real-time predictions.

---

## 🚀 Features

- **Text Preprocessing** — Lowercasing, stripping whitespace, removing nulls and duplicates
- **TF-IDF Vectorization** — Unigram + Bigram features with stop word removal
- **LinearSVC Classifier** — Fast and effective for high-dimensional text data
- **Confidence Score** — Uses SVM decision function to show how confident the prediction is
- **Model Persistence** — Saves trained model and vectorizer with `joblib` for reuse
- **Sample Testing** — Includes real-world style spam and ham test messages

---

## 🧠 How It Works

```
Raw Email Text
      ↓
Lowercase + Strip
      ↓
TF-IDF Vectorizer  (max 5000 features, unigrams + bigrams)
      ↓
LinearSVC Model
      ↓
Spam / Not Spam  +  Confidence Score
```

---

## 📊 Model Details

| Property | Value |
|---|---|
| Algorithm | LinearSVC |
| Vectorizer | TF-IDF |
| N-gram Range | (1, 2) — unigrams & bigrams |
| Max Features | 5,000 |
| Test Split | 20% |
| Metric | Accuracy + Classification Report |

---

## 🛠️ Requirements

```bash
pip install pandas scikit-learn joblib
```

---

## ▶️ How to Run

**1. Make sure you have `emails.csv` in the same directory** with two columns:
- `text` — the email body
- `spam` — label (`1` = spam, `0` = not spam)

**2. Run the script:**
```bash
python spam_detector.py
```

**3. Output:**
- Prints accuracy and classification report
- Tests sample messages and shows predictions
- Saves model files:
  - `svm_spam_model.pkl`
  - `tfidf_vectorizer.pkl`

---

## 🔁 Reusing the Saved Model

```python
import joblib

svm_model = joblib.load('svm_spam_model.pkl')
tfidf     = joblib.load('tfidf_vectorizer.pkl')

def predict(message):
    vec = tfidf.transform([message.lower().strip()])
    label = "Spam" if svm_model.predict(vec)[0] == 1 else "Not Spam"
    score = svm_model.decision_function(vec)[0]
    return f"{label}  (Score: {score:.4f})"

print(predict("You won a free iPhone! Claim now!"))
```

---

## 📂 Project Structure

```
spam-detector/
│
├── spam_detector.py        # Main training and prediction script
├── emails.csv              # Dataset (not included)
├── svm_spam_model.pkl      # Saved SVM model (generated after run)
└── tfidf_vectorizer.pkl    # Saved TF-IDF vectorizer (generated after run)
```

---

## ⚠️ Limitations

- Performance depends heavily on the quality and size of `emails.csv`
- No deep text normalization (stemming / lemmatization)
- LinearSVC does not output true probabilities (only decision scores)

---

## 💡 Possible Improvements

- [ ] Add **stemming/lemmatization** (NLTK or spaCy) for better text normalization
- [ ] Try **Logistic Regression** or **Naive Bayes** as baselines to compare
- [ ] Use **calibrated classifier** (`CalibratedClassifierCV`) for real probability output
- [ ] Build a simple **web interface** with Flask or Streamlit
- [ ] Cross-validate with **k-fold** instead of a single train/test split

---

## 📄 License

This project is open source and free to use.
