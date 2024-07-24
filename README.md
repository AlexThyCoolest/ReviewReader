# ReviewReader

## 📊 Analyze Amazon Product Reviews with AI

ReviewReader is an advanced Flask-based web application that leverages machine learning to analyze Amazon product reviews, helping users make informed purchasing decisions.

[![A 1 minute demo of the website](https://github.com/user-attachments/assets/dbb7f2c5-5d1f-4e3b-8bf9-38f25bd7c8f8)](https://github-production-user-asset-6210df.s3.amazonaws.com/118056225/351786228-dbb7f2c5-5d1f-4e3b-8bf9-38f25bd7c8f8.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240724%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240724T155535Z&X-Amz-Expires=300&X-Amz-Signature=ae1b9cc088fbfbdbeda5e39f7dd02af67f2b761098259232b168c412ec46fb67&X-Amz-SignedHeaders=host&actor_id=118056225&key_id=0&repo_id=832837105)

## 🌟 Features

- 🔍 Scrape and analyze Amazon product reviews in real-time
- 🤖 Use AI to distinguish between genuine and potentially fake reviews
- 📈 Provide sentiment analysis and product recommendations
- 🎭 Implement user agent rotation to avoid detection
- 🚀 Asynchronous processing for faster results

## 🛠️ Technologies Used

- **Backend**: Flask, Python 3.x
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: scikit-learn, NLTK, TextBlob
- **Web Scraping**: Selenium
- **Data Processing**: Pandas, NumPy

## 📋 Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7+
- pip
- virtualenv (recommended)
- Chrome browser (for Selenium WebDriver)

## 🚀 Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/ReviewReader.git
   cd ReviewReader
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Set up the Chrome WebDriver**

   Ensure you have the Chrome browser installed, then download the appropriate [ChromeDriver](https://sites.google.com/a/chromium.org/chromedriver/downloads) for your system and add it to your PATH.

4. **Run the application**

   ```bash
   python app.py
   ```

   The application will be available at `http://localhost:2020`.

## 💻 Usage

1. Open your web browser and navigate to `http://localhost:2020`.
2. Enter the name of an Amazon product you want to analyze.
3. Wait for the analysis to complete (you'll be entertained with jokes during this time!).
4. Review the results, including:
   - Overall product rating
   - Ratio of genuine to potentially fake reviews
   - Sentiment analysis
   - Purchase recommendation

## 📁 Project Structure

```
├── app.py                 # Main Flask application 
├── README.md              # Project documentation
├── static/                # Static files (CSS, JS, images)
│   ├── css/
│   ├── js/
│   └── img/
├── templates/             # HTML templates
│   ├── index.html
│   ├── loading.html
│   └── results.html
└── model/                # Pre-trained ML models and data
    ├── model.pkl
    ├── oprimal_threshold.pkl
    └── vectorizer.pkl
```

## 🧠 How It Works

1. **Web Scraping**: Uses Selenium to scrape Amazon product reviews.
2. **Data Preprocessing**: Cleans and tokenizes review text using NLTK.
3. **Feature Extraction**: Utilizes TF-IDF vectorization for text features.
4. **Machine Learning**: Employs Logistic Regression to classify reviews.
5. **Sentiment Analysis**: Analyzes review sentiment using TextBlob.
6. **Result Aggregation**: Combines ML predictions and sentiment analysis for final recommendation.

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 89% |
| Precision | 90% |
| Recall | 0.94 |
| F1 Score | 0.91 |

## 🤝 Contributing

Contributions to ReviewReader are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Contact

Alex Agboola - [My Portfolio](https://alexagboola.com) - alexagboolacodes@gmail.com

Project Link: [https://github.com/AlexThyCoolest/ReviewReader](https://github.com/AlexThyCoolest/ReviewReader)

## 🙏 Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)
- [Selenium](https://www.selenium.dev/)
- [TextBlob](https://textblob.readthedocs.io/)
