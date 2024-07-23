import time
from flask import Flask, render_template, request, redirect, url_for
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from fake_useragent import UserAgent
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re
import nltk
from textblob import TextBlob
from urllib.parse import unquote
import pickle

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    if isinstance(text, str):
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token not in string.punctuation]
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(tokens)
    else:
        return ""

# Function to get product details and reviews from Amazon
def get_product_details(product_name, max_reviews=1000):
    # ... (keep the existing get_product_details function as is)

# Function to classify reviews and calculate the ratio
def classify_and_calculate_ratio(reviews, model, vectorizer, threshold):
    # ... (keep the existing classify_and_calculate_ratio function as is)

# Function to analyze sentiment and provide recommendation
def analyze_sentiment_and_recommend(real_review_texts):
    # ... (keep the existing analyze_sentiment_and_recommend function as is)

# Function to load pre-trained model, vectorizer, and threshold
def load_model(model_path, vectorizer_path, threshold_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(threshold_path, 'rb') as f:
        optimal_threshold = pickle.load(f)
    return model, vectorizer, optimal_threshold

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/loading", methods=["POST"])
def loading():
    import random
    jokes = [
        # ... (keep the existing jokes list)
    ]
    product_name = request.form.get("product_name")
    random_jokes = random.sample(jokes, len(jokes))
    return render_template("loading.html", product_name=product_name, jokes=random_jokes)

@app.route("/results/<path:product_name>", methods=["GET"])
def results(product_name):
    product_name = unquote(product_name)
    max_reviews = 1000
    
    product_details = get_product_details(product_name, max_reviews=max_reviews)
    
    if product_details:
        # Load pre-trained model, vectorizer, and optimal threshold
        model, vectorizer, optimal_threshold = load_model('model.pkl', 'vectorizer.pkl', 'optimal_threshold.pkl')

        # Classify reviews and calculate statistics
        ratio, fake_count, real_count, avg_real_rating, real_review_texts = classify_and_calculate_ratio(product_details['Sample Reviews'], model, vectorizer, optimal_threshold)
        
        recommendation = analyze_sentiment_and_recommend(real_review_texts)
        
        return render_template("results.html", 
                               product_details=product_details, 
                               ratio=ratio, 
                               fake_count=fake_count, 
                               real_count=real_count, 
                               avg_real_rating=avg_real_rating,
                               recommendation=recommendation)
    else:
        return render_template("results.html", product_details=None)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2020, debug=True)