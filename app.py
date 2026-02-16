import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data (safe for Streamlit cloud/local)
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Load stopwords once (optimization)
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove special characters
    text = [word for word in text if word.isalnum()]

    # Remove stopwords & punctuation
    text = [word for word in text if word not in stop_words and word not in string.punctuation]

    # Stemming
    text = [ps.stem(word) for word in text]

    return " ".join(text)


# Load saved vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize (IMPORTANT: must be already fitted)
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display
        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")
