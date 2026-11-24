# Smart Fake News Detection System

## About the Project
Smart Fake News Detection System is a Machine Learning-based web application that classifies news articles as Likely Fake News or Likely Real News.

It uses Python, Natural Language Processing (NLP), TF-IDF and Machine Learning algorithms.

## Technologies Used
- Python
- Machine Learning
- Natural Language Processing (NLP)
- TF-IDF
- Logistic Regression
- Multinomial Naive Bayes
- Streamlit
- Scikit-learn
- NLTK

## Dataset
WELFake Dataset

- Original records: 72,134
- Cleaned records: 63,121
- Fake News: 34,791
- Real News: 28,330

## Machine Learning Models
1. Logistic Regression
2. Multinomial Naive Bayes

## Model Performance

| Model | Accuracy |
|---|---:|
| Logistic Regression | 96.19% |
| Multinomial Naive Bayes | 88.54% |

Logistic Regression was selected as the final model.

## Features
- Enter news title and content.
- Predict likely Fake or Real News.
- Display model confidence.
- Display Fake and Real probabilities.
- View cleaned text.

## How to Run the Project

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

## Project Files
- `app.py` – Streamlit application
- `best_fake_news_model.pkl` – Trained Logistic Regression model
- `tfidf_vectorizer.pkl` – TF-IDF vectorizer
- `requirements.txt` – Required Python libraries

## Note
The prediction is based on patterns learned from the training dataset. The system does not independently verify news facts.

## Author
Tannu257
