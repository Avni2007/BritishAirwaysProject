# British Airways AI Customer Satisfaction Project

## Project Overview

This project focuses on analyzing British Airways customer reviews using Artificial Intelligence (AI) and Natural Language Processing (NLP). The system predicts customer sentiment, identifies common complaint topics, generates personalized recommendations, and visualizes insights through an interactive dashboard.

---

## Objectives

* Perform sentiment analysis on customer reviews.
* Predict customer satisfaction levels.
* Identify common complaint topics.
* Generate personalized recommendations.
* Visualize customer insights using a dashboard.

---

## Dataset

Dataset Used: **British Airways Customer Reviews Dataset**

Columns:

* `title` – Title of customer review
* `reviews` – Complete customer review text

---

## Technologies Used

* Python
* Pandas
* VADER Sentiment Analysis
* Scikit-Learn
* Streamlit
* Matplotlib

---

## Features

### 1. Data Preprocessing

* Load customer review dataset
* Remove unnecessary columns
* Handle missing values

### 2. Sentiment Analysis

The project uses **VADER Sentiment Analyzer** to classify reviews into:

* Positive
* Negative
* Neutral

A sentiment score is also generated for each review.

### 3. Customer Satisfaction Prediction

A **Random Forest Classifier** predicts customer satisfaction (CSAT) using sentiment scores.

CSAT Mapping:

| Sentiment | CSAT Score |
| --------- | ---------- |
| Positive  | 5          |
| Neutral   | 3          |
| Negative  | 1          |

### 4. Topic Detection

The system identifies major customer complaint categories:

* Flight Delay
* Baggage Issues
* Customer Service
* Seat Comfort
* Food Quality
* Other

### 5. Recommendation Engine

Personalized recommendations are generated based on detected topics.

Examples:

| Topic            | Recommendation                            |
| ---------------- | ----------------------------------------- |
| Flight Delay     | Offer vouchers and improve scheduling     |
| Baggage          | Provide baggage tracking and compensation |
| Customer Service | Improve staff training                    |
| Seat Comfort     | Upgrade seats and legroom                 |
| Food             | Improve meal quality                      |

### 6. Dashboard

An interactive Streamlit dashboard provides:

* Total Reviews
* Positive Reviews
* Negative Reviews
* Sentiment Distribution
* Topic Distribution
* Recommendations Table
* Predicted Customer Satisfaction

---

## Project Structure

```text
BritishAirwaysProject

├── british_airways_reviews.csv
├── Final_British_Airways_Analysis.csv
├── main.py
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

Install dependencies:

```bash
pip install pandas
pip install vaderSentiment
pip install scikit-learn
pip install streamlit
pip install matplotlib
```

Or:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Run Main Program

```bash
python main.py
```

This will:

* Analyze customer reviews
* Predict sentiment
* Detect topics
* Generate recommendations
* Save results to:

```text
Final_British_Airways_Analysis.csv
```

### Run Dashboard

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser.

---

## Sample Output

### Sentiment Distribution

* Positive Reviews: 645
* Negative Reviews: 636
* Neutral Reviews: 19

### Topic Distribution

* Flight Delay
* Customer Service
* Food
* Baggage
* Seat Comfort
* Other

---

## Future Enhancements

* Implement BERTopic or LDA Topic Modeling
* Deploy model using AWS SageMaker
* Create REST API using Flask or FastAPI
* Deploy dashboard to cloud

---

## Author

**Avni**

British Airways AI-Driven Predictive Customer Satisfaction and Personalized Experience Recommendation System
