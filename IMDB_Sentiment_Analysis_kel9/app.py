import streamlit as st
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

nltk.download('stopwords')

# Preprocessing function
def preprocess_text(text):
    """Membersihkan teks: lowercase, hapus HTML/URL/tanda baca, stopwords, dan stemming"""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)          # Hapus HTML tags
    text = re.sub(r'http\S+', '', text)        # Hapus URL
    text = re.sub(r'[^a-zA-Z\s]', '', text)    # Hapus non-alphabet
    text = re.sub(r'\s+', ' ', text).strip()   # Hapus spasi berlebih

    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]  # Stopwords
    words = [PorterStemmer().stem(word) for word in words]                     # Stemming
    return ' '.join(words)


# Load model dan tokenizer
save_dir = r"C:\Python_project\IMDB_Sentiment_Analysis_kel9\bert_imdb_model"  # Path penyimpanan model
model = TFDistilBertForSequenceClassification.from_pretrained(save_dir)
tokenizer = DistilBertTokenizer.from_pretrained(save_dir)

# Streamlit UI
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("🎭 IMDB Sentiment Analysis")
st.write("Masukkan review film yang baru kamu selesai nonton, dan biarkan model menilai apakah itu positif atau negatif.")

user_input = st.text_area("Masukkan Review mu disini 👇", height=150)

if st.button("Analisis Sentimen"):
    if user_input.strip() == "":
        st.warning("Review kamu kosong, kamu pasti lupa yah mau nulis apa? 🤔")
    else:
        cleaned_text = preprocess_text(user_input)
        inputs = tokenizer(cleaned_text, return_tensors="tf", truncation=True, padding=True, max_length=256)
        outputs = model(inputs)
        probs = tf.nn.softmax(outputs.logits, axis=-1)
        label = tf.argmax(probs, axis=1).numpy()[0]
        confidence = float(tf.reduce_max(probs)) 
        sentiment = "positif 👍" if label == 1 else "negatif 👎"
        
        st.success(f"Hasil sentimen: **{sentiment}**")
        st.markdown(f"🧠 Tingkat keyakinan model: **{confidence:.2%}**")

        # feedback
        if sentiment.lower().startswith("positif 👍"):
            st.markdown("Film ini kelihatannya keren! 😎👌")
        else:
            st.markdown("Mungkin film ini kurang cocok buat kamu? 😔🙏")
