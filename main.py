import streamlit as st
import feedparser
from transformers import pipeline

# Try different models here, just go to huggingface and search for financial sentimental analysis model
pipe = pipeline("text-classification", model="ProsusAI/finbert")

st.title("Stock Sentiment Analysis App")
st.write("Analyze sentiment of news articles related to a stock.")

ticker = st.text_input("Enter Stock Ticker (e.g., META):", "META")
keyword = st.text_input("Enter Keyword for Filtering Articles:", ticker.lower())

if st.button("Analyze"):
    rss_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    feed = feedparser.parse(rss_url)

    total_score = 0
    num_articles = 0

    st.subheader("Articles and Sentiment Analysis")

    for entry in feed.entries:
        if keyword.lower() not in entry.summary.lower():
            continue

        sentiment = pipe(entry.summary)[0]
        label = sentiment["label"]
        score = sentiment["score"]

        st.write(f"**Title:** {entry.title}")
        st.write(f"**Link:** [Read Article]({entry.link})")
        st.write(f"**Published:** {entry.published}")
        st.write(f"**Summary:** {entry.summary}")
        st.write(f"**Sentiment:** {label}, **Score:** {score:.2f}")
        st.write("---")

        if label == "positive":
            total_score += score
            num_articles += 1
        elif label == "negative":
            total_score -= score
            num_articles += 1

    if num_articles > 0:
        overall_sentiment = "Positive" if total_score > 0.15 else "Negative" if total_score < -0.15 else "Neutral"
        st.subheader("Overall Sentiment")
        st.write(f"**Sentiment:** {overall_sentiment}")
        st.write(f"**Score:** {total_score:.2f}")
    else:
        st.write("No articles found matching the keyword.")
