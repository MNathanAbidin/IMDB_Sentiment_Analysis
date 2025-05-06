import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
import re

nltk.download('stopwords')

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

def predict_sentiment(text, model, tokenizer):
    """Prediksi sentimen dari teks input"""
    cleaned_text = preprocess_text(text)  # Preprocess konsisten
    inputs = tokenizer(
        cleaned_text,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=256
    )
    outputs = model(inputs)
    probs = tf.nn.softmax(outputs.logits, axis=-1)
    label = tf.argmax(probs, axis=1).numpy()[0]
    return "positive" if label == 1 else "negative"

save_dir = r"C:\Python_project\IMDB_Sentiment_Analysis_kel9\bert_imdb_model"

# Contoh penggunaan
loaded_model = TFDistilBertForSequenceClassification.from_pretrained(save_dir)
loaded_tokenizer = DistilBertTokenizer.from_pretrained(save_dir)

# Input dari pengguna
while True:
  user_input = input("Masukkan teks untuk dianalisis sentimen: ")
  result = predict_sentiment(user_input, loaded_model, loaded_tokenizer)
  print(f"Sentimen: {result}")
  choice = input("Ingin mencoba lagi? (y/n): ")
  if choice.lower() != 'y':
    break