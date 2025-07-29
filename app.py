import streamlit as st
import pickle
import nltk
import pytesseract
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Download stopwords if needed
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Optional: Set path to Tesseract executable (Windows only)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Preprocess text
def preprocess_text(text):
    tokens = text.lower().split()
    filtered = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered)

# Predict sentiment from text
def predict_sentiment(text):
    cleaned = preprocess_text(text)
    vector = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vector)
    return "Positive" if prediction[0] == 1 else "Negative"

# Extract text from image (fixed)
def extract_text_from_image(uploaded_file):
    try:
        pil_image = Image.open(uploaded_file)  # ✅ Convert to PIL.Image
        text = pytesseract.image_to_string(pil_image)
        return text.strip()
    except Exception as e:
        return f"[Error reading image] {e}"

# Predict sentiment from image
def predict_sentiment_from_image(uploaded_file):
    extracted_text = extract_text_from_image(uploaded_file)
    if not extracted_text or extracted_text.startswith("[Error"):
        return "Error", extracted_text
    sentiment = predict_sentiment(extracted_text)
    return sentiment, extracted_text

# Streamlit UI
st.set_page_config(page_title="🎭 Sentiment Analyzer", layout="centered")
st.title("🎭 Movie Review Sentiment Analyzer")
st.markdown("Enter a movie review manually or upload a review image to predict whether it's **Positive** or **Negative**.")

# TEXT REVIEW
st.subheader("📝 Text Review")
user_review = st.text_area("Enter your review here:")

if st.button("🔍 Analyze Text"):
    if user_review.strip() == "":
        st.warning("Please enter a review before analyzing.")
    else:
        sentiment = predict_sentiment(user_review)
        color = "#014d20" if sentiment == "Positive" else "#6e0000"
        emoji = "😊" if sentiment == "Positive" else "😠"
        st.markdown(
            f"""
            <div style='padding: 15px; border-radius: 10px; background-color: {color}; color: white; font-size: 18px; text-align: center;'>
                Predicted Sentiment: {emoji} <strong>{sentiment}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

# IMAGE REVIEW
st.subheader("📸 Image Review")
uploaded_image = st.file_uploader("Upload an image containing a review", type=["png", "jpg", "jpeg"])

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Review Image", use_container_width=True)

    if st.button("🧠 Analyze Image"):
        sentiment, extracted_text = predict_sentiment_from_image(uploaded_image)

        st.markdown("**🧾 Extracted Text from Image:**")
        st.code(extracted_text)

        if sentiment == "Error":
            st.error("❌ Failed to extract text from image.")
        else:
            color = "#014d20" if sentiment == "Positive" else "#6e0000"
            emoji = "😊" if sentiment == "Positive" else "😠"
            st.markdown(
                f"""
                <div style='padding: 15px; border-radius: 10px; background-color: {color}; color: white; font-size: 18px; text-align: center;'>
                    Image Sentiment: {emoji} <strong>{sentiment}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )
