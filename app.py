import streamlit as st
import pandas as pd


df = pd.read_csv(

"Final_British_Airways_Analysis.csv"

)


st.title(

"British Airways AI Customer Satisfaction Dashboard"

)


st.metric(

"Total Reviews",

len(df)

)


st.metric(

"Positive Reviews",

len(

df[

df["Sentiment"]=="Positive"

]

)

)


st.metric(

"Negative Reviews",

len(

df[

df["Sentiment"]=="Negative"

]

)

)



st.subheader(

"Sentiment Distribution"

)

st.bar_chart(

df["Sentiment"].value_counts()

)



st.subheader(

"Topic Distribution"

)

st.bar_chart(

df["Topic"].value_counts()

)



st.subheader(

"Recommendations"

)

st.dataframe(

df[

[

"title",

"Sentiment",

"Topic",

"Recommendation",

"CSAT"

]

]

)