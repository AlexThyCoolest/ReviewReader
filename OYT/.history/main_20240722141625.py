import time
from flask import Flask, render_template, request
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from fake_useragent import UserAgent
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re
import nltk

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
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, '[data-component-type="s-search-result"]')))
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        selectors = [
            'div[data-component-type="s-search-result"] h2 a',
            '.s-result-item h2 a',
            '.s-search-results .s-result-item a.a-link-normal',
        ]
        
        first_product = None
        for selector in selectors:
            try:
                first_product = driver.find_element(By.CSS_SELECTOR, selector)
                break
            except NoSuchElementException:
                continue
        
        if not first_product:
            print("Could not find the product link.")
            return {}
        
        first_product.click()
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "productTitle")))
        
        title = driver.find_element(By.ID, "productTitle").text
        author = driver.find_element(By.CSS_SELECTOR, '.author .contributorNameID').text if driver.find_elements(By.CSS_SELECTOR, '.author .contributorNameID') else "N/A"
        rating = driver.find_element(By.CSS_SELECTOR, 'span[data-hook="rating-out-of-text"]').text if driver.find_elements(By.CSS_SELECTOR, 'span[data-hook="rating-out-of-text"]') else "N/A"
        reviews_count = driver.find_element(By.CSS_SELECTOR, 'span[data-hook="total-review-count"]').text if driver.find_elements(By.CSS_SELECTOR, 'span[data-hook="total-review-count"]') else "N/A"
        
        reviews_link = driver.find_element(By.CSS_SELECTOR, 'a[data-hook="see-all-reviews-link-foot"]')
        reviews_link.click()
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "cm_cr-review_list")))
        
        reviews = []
        while len(reviews) < max_reviews:
            review_elements = driver.find_elements(By.CSS_SELECTOR, 'div[data-hook="review"]')
            for review in review_elements:
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
            
            next_page = driver.find_elements(By.CSS_SELECTOR, 'li.a-last a')
            if next_page and len(reviews) < max_reviews:
                next_page[0].click()
                time.sleep(2)
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
    for review in reviews:
        preprocessed_text = preprocess_text(review['Text'])
        vectorized_text = vectorizer.transform([preprocessed_text])
        probability = model.predict_proba(vectorized_text)[0, 1]
        prediction = 1 if probability >= threshold else 0
        if prediction == 1:
            fake_count += 1
        else:
            real_count += 1
            try:
                rating_text = review['Rating']
                match = re.search(r'(\d+(\.\d+)?)', rating_text)
                if match:
                    rating = float(match.group(1))
                    real_ratings.append(rating)
                else:
                    print(f"Couldn't extract rating from: {rating_text}")
            except Exception as e:
                print(f"Error processing rating: {review['Rating']}. Error: {str(e)}")
    
    ratio = f"{fake_count}:{real_count}"
    avg_real_rating = sum(real_ratings) / len(real_ratings) if real_ratings else 0
    return ratio, fake_count, real_count, avg_real_rating

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/results", methods=["POST"])
def results():
    product_name = request.form.get("product_name")
    max_reviews = 100
    
    product_details = get_product_details(product_name, max_reviews=max_reviews)
    
    if product_details:
        # Load and preprocess the dataset
        df = pd.read_csv("amazon_reviews.csv")
        df['text_'] = df['text_'].fillna('')
        df = df.dropna(subset=['label', 'text_'])
        label_map = {'CG': 0, 'OR': 1}
        df['label_encoded'] = df['label'].map(label_map)
        df = df.dropna(subset=['label_encoded'])
        X = df['text_']
        y = df['label_encoded'].astype(int)

        # Split the data and train the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)

        model = LogisticRegression(random_state=42, class_weight='balanced')
        model.fit(X_train_vectorized, y_train)

        # Find optimal threshold
        y_pred_proba = model.predict_proba(X_test_vectorized)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        # Classify reviews and calculate statistics
        ratio, fake_count, real_count, avg_real_rating = classify_and_calculate_ratio(product_details['Sample Reviews'], model, vectorizer, optimal_threshold)
        
        return render_template("results.html", 
                               product_details=product_details, 
                               ratio=ratio, 
                               fake_count=fake_count, 
                               real_count=real_count, 
                               avg_real_rating=avg_real_rating)
    else:
        return render_template("results.html", product_details=None)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2020, debug=True)