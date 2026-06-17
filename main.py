import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("british_airways_reviews.csv")

# Remove unwanted column
df.drop("Unnamed: 0", axis=1, inplace=True)

print("Dataset Loaded Successfully\n")

print(df.head())


# ==========================
# SENTIMENT ANALYSIS
# ==========================

analyzer = SentimentIntensityAnalyzer()


def get_sentiment(review):

    score = analyzer.polarity_scores(str(review))

    compound = score['compound']

    if compound >= 0.05:

        return "Positive"

    elif compound <= -0.05:

        return "Negative"

    else:

        return "Neutral"


def get_score(review):

    return analyzer.polarity_scores(str(review))['compound']


df["Sentiment"] = df["reviews"].apply(get_sentiment)

df["Score"] = df["reviews"].apply(get_score)


print("\nSentiment Count:\n")

print(df["Sentiment"].value_counts())


# ==========================
# TOPIC DETECTION
# ==========================

topics = {

    "Flight Delay":
    ["delay","delayed","late","waiting"],

    "Baggage":
    ["baggage","luggage","bag"],

    "Customer Service":
    ["staff","service","crew","rude"],

    "Seat Comfort":
    ["seat","legroom","comfort"],

    "Food":
    ["food","meal","drink"]

}


def get_topic(review):

    review = str(review).lower()

    for topic, words in topics.items():

        for word in words:

            if word in review:

                return topic

    return "Other"


df["Topic"] = df["reviews"].apply(get_topic)


print("\nTopic Count:\n")

print(df["Topic"].value_counts())


# ==========================
# RECOMMENDATIONS
# ==========================

recommendations = {

    "Flight Delay":
    "Offer vouchers and improve scheduling",

    "Baggage":
    "Provide baggage tracking and compensation",

    "Customer Service":
    "Improve staff training and response time",

    "Seat Comfort":
    "Upgrade seats and increase legroom",

    "Food":
    "Improve meal quality and menu options",

    "Other":
    "Further investigation required"

}


df["Recommendation"] = df["Topic"].map(recommendations)


print("\nSample Recommendations:\n")

print(

df[

['Topic',

'Recommendation']

].head(10)

)


# ==========================
# CUSTOMER SATISFACTION PREDICTION
# ==========================

mapping = {

    "Positive":5,

    "Neutral":3,

    "Negative":1

}


df["CSAT"] = df["Sentiment"].map(mapping)


X = df[["Score"]]

y = df["CSAT"]


X_train, X_test, y_train, y_test = train_test_split(

X,

y,

test_size=0.2,

random_state=42

)


model = RandomForestClassifier()

model.fit(X_train,y_train)


accuracy = model.score(

X_test,

y_test

)


print("\nPrediction Accuracy:")

print(round(accuracy*100,2),"%")



# ==========================
# SAVE FINAL FILE
# ==========================

df.to_csv(

"Final_British_Airways_Analysis.csv",

index=False

)


print("\nFinal file saved successfully!")

print("File Name: Final_British_Airways_Analysis.csv")