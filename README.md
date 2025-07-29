# 🎭 Movie Review Sentiment Analyzer

A smart, interactive sentiment analysis tool that can classify **movie reviews** as either **Positive** or **Negative** using:
- ✅ Manual text input
- 📸 OCR-based image review input (via Tesseract)

Built using:
- Python 🐍
- Streamlit for the frontend
- Logistic Regression with TF-IDF vectorization for the ML model

---

## 📁 Folder Structure

sentiment-analyzer/
│
├── app.py # Streamlit frontend for live prediction
├── train_sentiment_model.py # Script to train and save model/vectorizer
├── sentiment_app_full.py # (Optional) Combined training + app script
├── model.pkl # Saved logistic regression model
├── vectorizer.pkl # Saved TF-IDF vectorizer
├── IMDB Dataset.csv # Dataset for training the model
├── sample_review_image.png # Example image for testing OCR
└── README.md # Project documentation (you're here)



---

## 🚀 Features

- 🧠 Logistic Regression classifier trained on IMDB reviews
- ✍️ Accepts manual text input
- 📸 OCR support for reading reviews from uploaded images
- 🎨 Color-coded sentiment output (green for positive, red for negative)
- ⚡ Fast prediction using pre-trained model and vectorizer
- 📦 Lightweight and easy to deploy

---

## 🧠 Model Training

Run the following script to train your model:
```bash
python sentiment.py


This will:

Load the IMDB dataset

Clean and preprocess reviews

Train a Logistic Regression model with TF-IDF features

Save the model to model.pkl

Save the vectorizer to vectorizer.pkl

1. Clone the Repo or Download

git clone https://github.com/your-username/sentiment-analyzer.git
cd sentiment-analyzer

2. Install Requirements

pip install -r requirements.txt


Tesseract OCR Setup (for Image Reviews)


