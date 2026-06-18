import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("Final_British_Airways_Analysis.csv")

st.title("British Airways Review Analysis Dashboard")

# ==========================
# METRICS
# ==========================

st.subheader("Overview")

st.write("Total Reviews:", len(df))
st.write("Positive Reviews:", len(df[df["Sentiment"] == "Positive"]))
st.write("Negative Reviews:", len(df[df["Sentiment"] == "Negative"]))
st.write("Neutral Reviews:", len(df[df["Sentiment"] == "Neutral"]))

# ==========================
# SENTIMENT GRAPH
# ==========================

st.subheader("Sentiment Analysis")

sentiment_counts = df["Sentiment"].value_counts()

fig1, ax1 = plt.subplots()
ax1.bar(sentiment_counts.index, sentiment_counts.values)
ax1.set_title("Sentiment Distribution")
st.pyplot(fig1)

# ==========================
# TOPIC GRAPH
# ==========================

st.subheader("Complaint Topics")

topic_counts = df["Topic"].value_counts()

fig2, ax2 = plt.subplots()
ax2.bar(topic_counts.index, topic_counts.values)
ax2.set_title("Topic Distribution")
plt.xticks(rotation=20)
st.pyplot(fig2)

# ==========================
# SAMPLE DATA
# ==========================

st.subheader("Sample Data")
st.dataframe(df.head(10))

# ==========================
# RECOMMENDATIONS
# ==========================

st.subheader("Recommendations")

st.dataframe(df[['Topic', 'Recommendation']].drop_duplicates())