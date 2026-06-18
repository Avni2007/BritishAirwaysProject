# ==========================
# IMPORTS
# ==========================

import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ==========================
# DOWNLOAD NLTK RESOURCES
# ==========================

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("british_airways_reviews.csv")

if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

print("Dataset Loaded Successfully")
print(df.head())

# ==========================
# NLP PREPROCESSING
# ==========================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)

df["clean_reviews"] = df["reviews"].apply(clean_text)

# ==========================
# SENTIMENT ANALYSIS (VADER)
# ==========================

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))['compound']

    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["clean_reviews"].apply(get_sentiment)

df["Positive"] = df["clean_reviews"].apply(lambda x: analyzer.polarity_scores(x)['pos'])
df["Negative"] = df["clean_reviews"].apply(lambda x: analyzer.polarity_scores(x)['neg'])
df["Neutral"] = df["clean_reviews"].apply(lambda x: analyzer.polarity_scores(x)['neu'])

print("\nSentiment Counts:")
print(df["Sentiment"].value_counts())

# ==========================
# LDA TOPIC MODELING (REQUIRED)
# ==========================

vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(df["clean_reviews"])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(dtm)

topic_results = lda.transform(dtm)
df["Topic"] = topic_results.argmax(axis=1)

# Topic labels
topic_labels = {
    0: "Flight Issues",
    1: "Baggage Problems",
    2: "Customer Service",
    3: "Food Quality",
    4: "Seat Comfort"
}

df["Topic"] = df["Topic"].map(topic_labels)

print("\nTopic Counts:")
print(df["Topic"].value_counts())

# ==========================
# RECOMMENDATION ENGINE
# ==========================

recommendations = {
    "Flight Issues": "Improve scheduling and reduce delays",
    "Baggage Problems": "Enhance baggage tracking system",
    "Customer Service": "Train staff for better service",
    "Food Quality": "Improve in-flight meals",
    "Seat Comfort": "Upgrade seating comfort"
}

df["Recommendation"] = df["Topic"].map(recommendations)

# ==========================
# CSAT PREDICTION MODEL
# ==========================

mapping = {
    "Positive": 5,
    "Neutral": 3,
    "Negative": 1
}

df["CSAT"] = df["Sentiment"].map(mapping)

X = df[["Positive", "Negative", "Neutral"]]
y = df["CSAT"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nModel Accuracy:")
print(accuracy_score(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))

# ==========================
# VISUALIZATION 1 - SENTIMENT
# ==========================

sentiment_counts = df["Sentiment"].value_counts()

plt.figure()
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.title("Customer Sentiment Analysis")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.savefig("sentiment_graph.png")
plt.show()

# ==========================
# VISUALIZATION 2 - TOPIC
# ==========================

topic_counts = df["Topic"].value_counts()

plt.figure()
plt.bar(topic_counts.index, topic_counts.values)
plt.title("Customer Complaint Topics (LDA)")
plt.xlabel("Topic")
plt.ylabel("Count")
plt.xticks(rotation=20)
plt.savefig("topic_graph.png")
plt.show()

# ==========================
# SAVE OUTPUT
# ==========================

df.to_csv("Final_BA_Analysis.csv", index=False)

print("\nFiles generated:")
print("- sentiment_graph.png")
print("- topic_graph.png")
print("- Final_BA_Analysis.csv")