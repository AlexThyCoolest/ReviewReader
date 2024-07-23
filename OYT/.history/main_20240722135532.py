from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import joblib
import os
import random
import pickle
from textblob import TextBlob
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from fake_useragent import UserAgent

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

app = Flask(__name__)

# Function to load model and vectorizer
def load_model(model_path, vectorizer_path, threshold_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    with open(threshold_path, 'rb') as threshold_file:
        optimal_threshold = pickle.load(threshold_file)
    return model, vectorizer, optimal_threshold

# Function to scrape article content from URL
def scrape_article(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Try multiple methods to extract content
        content = ""

        # Method 1: Look for common article containers
        for container in [
            "article",
            "main",
            'div[class*="content"]',
            'div[class*="article"]',
            'div[id*="content"]',
            'div[id*="article"]',
        ]:
            if content:
                break
            elements = soup.select(container)
            if elements:
                content = elements[0].get_text(strip=True, separator=" ")

        # Method 2: If still empty, get all paragraph text
        if not content:
            paragraphs = soup.find_all("p")
            content = " ".join([p.get_text(strip=True) for p in paragraphs])

        # Method 3: If still empty, get all div text
        if not content:
            divs = soup.find_all("div")
            content = " ".join([div.get_text(strip=True) for div in divs])

        # Final check
        if content:
            return clean_text(content)
        else:
            print("Couldn't extract meaningful content from the webpage.")
            return None
    except requests.RequestException as e:
        print(f"An error occurred while fetching the article: {e}")
        return None

# Function to clean text
def clean_text(text):
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove special characters, keeping basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    return text.strip()

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)

# Function to predict fake news
def predict_fake_news(text, model, vectorizer):
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]

# Load model and vectorizer
model, vectorizer, optimal_threshold = load_model("model/model.pkl", "model/vectorizer.pkl", "model/optimal_threshold.pkl")

# Function to get product details and reviews from Amazon
def get_product_details(product_name, product_position, max_reviews=10000):
    ua = UserAgent()
    user_agent = ua.random

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(f"user-agent={user_agent}")

    driver = webdriver.Chrome(options=chrome_options)

    try:
        search_url = f"https://www.amazon.com/s?k={product_name.replace(' ', '+')}"
        driver.get(search_url)

        # Wait for the search results to load
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, '[data-component-type="s-search-result"]')))

        # Scroll to the bottom of the page
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.uniform(2, 4))  # Random wait time

        # Try different selectors for the products
        selectors = [
            'div[data-component-type="s-search-result"] h2 a',
            '.s-result-item h2 a',
            '.s-search-results .s-result-item a.a-link-normal',
        ]

        product_links = []
        for selector in selectors:
            product_links = driver.find_elements(By.CSS_SELECTOR, selector)
            if product_links:
                break

        if not product_links:
            print("Could not find any product links.")
            return {}

        if product_position > len(product_links):
            print(f"Product position {product_position} is out of range. Only {len(product_links)} products found.")
            return {}

        selected_product = product_links[product_position - 1]
        selected_product.click()

        # Wait for the product page to load
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "productTitle")))

        title = driver.find_element(By.ID, "productTitle").text
        author = driver.find_element(By.CSS_SELECTOR, '.author .contributorNameID').text if driver.find_elements(By.CSS_SELECTOR, '.author .contributorNameID') else "N/A"
        rating = driver.find_element(By.CSS_SELECTOR, 'span[data-hook="rating-out-of-text"]').text if driver.find_elements(By.CSS_SELECTOR, 'span[data-hook="rating-out-of-text"]') else "N/A"
        reviews_count = driver.find_element(By.CSS_SELECTOR, 'span[data-hook="total-review-count"]').text if driver.find_elements(By.CSS_SELECTOR, 'span[data-hook="total-review-count"]') else "N/A"

        # Navigate to reviews page
        reviews_link = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[data-hook="see-all-reviews-link-foot"]'))
        )
        reviews_link.click()

        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "cm_cr-review_list")))

        reviews = []
        while len(reviews) < max_reviews:
            review_elements = driver.find_elements(By.CSS_SELECTOR, 'div[data-hook="review"]')
            for review in review_elements:
                try:
                    reviewer_name = review.find_element(By.CSS_SELECTOR, '.a-profile-name').text
                    try:
                        review_rating = review.find_element(By.CSS_SELECTOR, 'i[data-hook="review-star-rating"]').get_attribute('innerHTML').strip()
                    except NoSuchElementException:
                        try:
                            review_rating = review.find_element(By.CSS_SELECTOR, 'i[data-hook="cmps-review-star-rating"]').get_attribute('innerHTML').strip()
                        except NoSuchElementException:
                            review_rating = "Rating not found"
                    review_text = review.find_element(By.CSS_SELECTOR, 'span[data-hook="review-body"]').text.strip()
                    reviews.append({
                        'Reviewer': reviewer_name,
                        'Rating': review_rating,
                        'Text': review_text
                    })
                    if len(reviews) >= max_reviews:
                        break
                except StaleElementReferenceException:
                    continue

            # Check if there's a next page button and click it
            next_page = driver.find_elements(By.CSS_SELECTOR, 'li.a-last a')
            if next_page and len(reviews) < max_reviews:
                next_page[0].click()
                time.sleep(random.uniform(2, 4))  # Random wait time
                WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "cm_cr-review_list")))
            else:
                break

        return {
            'Title': title,
            'Author': author,
            'Overall Rating': rating,
            'Total Reviews': reviews_count,
            'Sample Reviews': reviews
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {}
    finally:
        driver.quit()

# Function to classify reviews and calculate the ratio
def classify_and_calculate_ratio(reviews, model, vectorizer, threshold):
    fake_count = 0
    real_count = 0
    real_ratings = []
    real_review_texts = []

    for review in reviews:
        preprocessed_text = preprocess_text(review['Text'])
        vectorized_text = vectorizer.transform([preprocessed_text])
        probability = model.predict_proba(vectorized_text)[0, 1]
        prediction = 1 if probability >= threshold else 0
        if prediction == 1:
            fake_count += 1
        else:
            real_count += 1
            real_review_texts.append(review['Text'])
            # Extract numeric rating from string
            try:
                rating_text = review['Rating']
                print(f"Processing rating: {rating_text}")  # Debug statement
                # Extract the numeric value from the rating text
                match = re.search(r'(\d+(\.\d+)?)', rating_text)
                if match:
                    rating = float(match.group(1))
                    real_ratings.append(rating)
                    print(f"Extracted rating: {rating}")  # Debug statement
                else:
                    print(f"Couldn't extract rating from: {rating_text}")
            except Exception as e:
                print(f"Error processing rating: {review['Rating']}. Error: {str(e)}")

    print(f"Real ratings: {real_ratings}")  # Debug statement
    ratio = f"{fake_count}:{real_count}"
    avg_real_rating = sum(real_ratings) / len(real_ratings) if real_ratings else 0
    return ratio, fake_count, real_count, avg_real_rating, real_review_texts

# Function to analyze sentiment and provide recommendation
def analyze_sentiment_and_recommend(real_review_texts):
    overall_sentiment = 0
    for review in real_review_texts:
        sentiment = TextBlob(review).sentiment.polarity
        overall_sentiment += sentiment

    avg_sentiment = overall_sentiment / len(real_review_texts) if real_review_texts else 0

    if avg_sentiment > 0.2:
        return "Based on the real reviews, I recommend buying this product. The overall sentiment is positive."
    elif avg_sentiment < -0.2:
        return "Based on the real reviews, I do not recommend buying this product. The overall sentiment is negative."
    else:
        return "Based on the real reviews, the sentiment is mixed. Consider reading more reviews before making a decision."

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/results", methods=["POST"])
def results():
    product_name = request.form.get("product_name")
    product_position = int(request.form.get("product_position"))

    product_details = get_product_details(product_name, product_position)

    if product_details:
        # Classify reviews and calculate statistics
        ratio, fake_count, real_count, avg_real_rating, real_review_texts = classify_and_calculate_ratio(product_details['Sample Reviews'], model, vectorizer, optimal_threshold)

        # Analyze sentiment and provide recommendation
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
    app.run(debug=True)