import pandas as pd
import nltk
import pickle
import pytesseract
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Download required NLTK data
nltk.download('stopwords')

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# ----- Text Preprocessing -----
def preprocess_text(text):
    tokens = text.lower().split()
    filtered = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered)

# ----- Load and Process Dataset -----
df = pd.read_csv("IMDB Dataset.csv")
df['clean_review'] = df['review'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['clean_review']).toarray()
y = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----- Train Models -----
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Linear SVM": LinearSVC()
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n{name} Results")
    print("Accuracy:", round(acc, 4))
    print("F1 Score:", round(f1, 4))
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
    results.append({"Model": name, "Accuracy": acc, "F1 Score": f1})

# ----- Select and Save Best Model -----
best_model = models["Logistic Regression"]
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("\n✅ Model and vectorizer saved as 'model.pkl' and 'vectorizer.pkl'.")

# ----- Predict Text Sentiment -----
def predict_sentiment(text, model=best_model):
    processed = preprocess_text(text)
    vector = vectorizer.transform([processed]).toarray()
    prediction = model.predict(vector)
    return "Positive" if prediction[0] == 1 else "Negative"

# ----- Image-Based Sentiment -----
# Tesseract path (for Windows only)
# Comment this line if Tesseract is already in PATH or you're on Linux/Mac
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"[Error reading image] {e}"

def predict_sentiment_from_image(image_path, model=best_model):
    text = extract_text_from_image(image_path)
    if not text:
        return "No text detected"
    return predict_sentiment(text, model)

# ----- Sample Usage -----
if __name__ == "__main__":
    sample_review = "The acting was brilliant and the story was deeply moving."
    print("\nSample Text Sentiment:")
    print("Review:", sample_review)
    print("Sentiment:", predict_sentiment(sample_review))

    sample_image = "sample_review_image.jpg"  # Make sure this image exists
    print("\nSample Image Sentiment:")
    print("Image Path:", sample_image)
    print("Sentiment:", predict_sentiment_from_image(sample_image))
