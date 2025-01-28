import streamlit as st
import feedparser
import requests
import pandas as pd
import plotly.graph_objs as go
import time
from transformers import pipeline
import yfinance as yf
from streamlit.components.v1 import html as st_html

# -----------------------------
# Alpha Vantage & Model Config
# -----------------------------
ALPHA_VANTAGE_API_KEY = st.secrets["APIKEY"]  # Replace with your Alpha Vantage API key
ALPHA_VANTAGE_API_URL = 'https://www.alphavantage.co/query'

MODEL_OPTIONS = {
    "FinBERT (ProsusAI/finbert)": "ProsusAI/finbert",
    "RoBERTa-Large English (siebert/sentiment-roberta-large-english)": "siebert/sentiment-roberta-large-english",
    "Twitter RoBERTa (cardiffnlp/twitter-roberta-base-sentiment)": "cardiffnlp/twitter-roberta-base-sentiment",
    "Multilingual BERT (nlptown/bert-base-multilingual-uncased-sentiment)": "nlptown/bert-base-multilingual-uncased-sentiment",
    "DistilBERT Emotion (distilbert-base-uncased-emotion)": "j-hartmann/emotion-english-distilroberta-base"
}

# -----------------------------
# Scrolling Ticker Functions
# -----------------------------
stock_tickers = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA',
    'JPM', 'JNJ', 'V', 'WMT', 'PG', 'XOM', 'HD', 'BAC',
    'DIS', 'NFLX', 'INTC', 'KO', 'PEP', 'CSCO', 'VZ',
    'MCD', 'ADBE', 'CMCSA'
]

def get_stock_prices(tickers):
    """
    Fetches the most recent closing price for each ticker from Yahoo Finance.
    """
    prices = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        todays_data = stock.history(period='1d')
        if not todays_data.empty:
            prices[ticker] = todays_data['Close'][0]
    return prices

def display_scrolling_ticker(prices):
    """
    Generates and displays an HTML-based ticker using CSS animations.
    """
    # Format the ticker text
    ticker_text = ' | '.join([f"{ticker}: ${price:.2f}" for ticker, price in prices.items()])

    # HTML + CSS for the scrolling effect
    html_code = f"""
    <div class="ticker-container">
      <div class="ticker-text">
        {ticker_text}
      </div>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

# -----------------------------
# Stock & Sentiment Functions
# -----------------------------
def fetch_stock_data(symbol):
    """
    Fetches intraday stock data from Alpha Vantage.
    """
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': symbol,
        'interval': '5min',
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(ALPHA_VANTAGE_API_URL, params=params)
    data = response.json()
    if 'Time Series (5min)' in data:
        df = pd.DataFrame.from_dict(data['Time Series (5min)'], orient='index', dtype=float)
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df
    else:
        st.error("Error fetching stock data. Please check the ticker symbol and try again.")
        return None

def plot_stock_data(df, symbol):
    """
    Plots the stock's closing price over time using Plotly.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    fig.update_layout(
        title=f'Real-Time Stock Price for {symbol}',
        xaxis_title='Time',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Main Streamlit App
# -----------------------------
def main():
    # Set a wide layout and give a custom page title
    st.set_page_config(page_title="Integrated Stock Sentiment App", layout="wide")

    # --- Inject Custom CSS for Modern Look ---
    st.markdown(
        """
        <style>
        /* Import a modern font from Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        /* Set up background gradient and font */
        html, body {
            background: linear-gradient(120deg, #1C1C1C, #2E2E2E);
            font-family: 'Poppins', sans-serif;
            color: #F5F5F5;
        }
        /* The main container that holds Streamlit elements */
        .block-container {
            background-color: transparent !important;
            padding: 2rem !important;
        }

        /* Headings color */
        h1, h2, h3, h4, h5 {
            color: #00FFC8; /* Bright accent color */
            margin-top: 1rem;
        }

        /* Scrolling Ticker Container */
        .ticker-container {
            background-image: linear-gradient(to right, #06beb6, #48b1bf);
            color: #fff;
            overflow: hidden;
            white-space: nowrap;
            box-sizing: border-box;
            padding: 10px 0;
            margin-bottom: 20px;
            border-radius: 10px;
        }
        .ticker-text {
            display: inline-block;
            padding-left: 100%;
            font-size: 1.2rem;
            font-weight: bold;
            animation: ticker 30s linear infinite;
        }
        @keyframes ticker {
          0%   { transform: translate3d(0, 0, 0); }
          100% { transform: translate3d(-100%, 0, 0); }
        }

        /* Buttons: for the Analyze button, etc. */
        .stButton>button {
            background-color: #00FFC8;
            color: #1C1C1C;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            font-weight: 600;
            transition: background-color 0.2s ease-in-out, color 0.2s ease-in-out;
            border: none;
        }
        .stButton>button:hover {
            background-color: #48b1bf;
            color: #fff;
            cursor: pointer;
        }

        /* Some generic "card" styling for sections */
        .content-card {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }

        /* For the "Overall Sentiment" text output */
        .sentiment-box {
            background-color: #1c1c1c;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
            color: #00FFC8;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --- A Simple Embedded React Component (Demo Purposes) ---
    # st.subheader("React Component Demo")
    # st.write("Below is an example of embedding a simple React component directly into Streamlit. It says hello with a React 'div':")
    # react_code = """
    # <div id="rootReact"></div>
    # <!-- Load React and ReactDOM (CDN) -->
    # <script src="https://cdnjs.cloudflare.com/ajax/libs/react/16.8.6/umd/react.development.js"></script>
    # <script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/16.8.6/umd/react-dom.development.js"></script>
    # <script>
    #   // A small React component
    #   const e = React.createElement;
    #   function HelloFromReact(props) {
    #     return e('div', { style: { color: '#00FFC8', fontSize: '1.2rem' }}, `Hello from React, ${props.name}!`);
    #   }

    #   ReactDOM.render(
    #     e(HelloFromReact, { name: 'Streamlit User' }, null),
    #     document.getElementById('rootReact')
    #   );
    # </script>
    # """
    # st_html(react_code, height=60)

    # --- Main Title ---
    st.title("Stock Sentiment Analysis App with Real-Time Scrolling Ticker")

    # --- Ticker Section ---
    st.write("### Live Stock Prices")
    with st.container():
        stock_prices = get_stock_prices(stock_tickers)
        display_scrolling_ticker(stock_prices)

    st.write("---")

    # --- User Inputs for Sentiment Analysis ---
    st.write("### Sentiment Analysis Controls")
    selected_model = st.selectbox("Choose a sentiment model", list(MODEL_OPTIONS.keys()))
    ticker = st.text_input("Enter Stock Ticker (e.g., META):", "META")
    keyword = st.text_input("Enter Keyword for Filtering Articles:", "meta")

    # --- Analysis Button ---
    if st.button("Analyze"):
        st.write("## Results")

        # 1. Load selected model pipeline
        model_name = MODEL_OPTIONS[selected_model]
        with st.spinner("Loading sentiment analysis model..."):
            pipe = pipeline("text-classification", model=model_name)

        # 2. Fetch stock data & display chart
        with st.spinner(f"Fetching stock data for {ticker}..."):
            stock_data = fetch_stock_data(ticker)
            if stock_data is not None:
                with st.container():
                    st.write(f"### Intraday Stock Chart for {ticker}")
                    plot_stock_data(stock_data, ticker)

        # 3. Fetch RSS feed & run sentiment
        rss_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
        with st.spinner("Fetching and analyzing news articles..."):
            feed = feedparser.parse(rss_url)

            total_score = 0
            num_articles = 0

            st.write("### Articles & Sentiment Analysis")
            with st.container():
                start_time = time.time()

                for entry in feed.entries:
                    # Filter based on keyword in title or summary
                    if keyword.lower() not in entry.summary.lower() and keyword.lower() not in entry.title.lower():
                        continue

                    sentiment = pipe(entry.summary)[0]
                    label = sentiment["label"]
                    score = sentiment["score"]

                    # Display each article's info in a 'card'
                    with st.expander(entry.title, expanded=False):
                        st.markdown(f"""**Link:** [Read Article]({entry.link})
**Published:** {entry.published}
**Summary:** {entry.summary}""")
                        st.write(f"**Sentiment:** {label}, **Score:** {score:.2f}")

                    # Basic aggregated score calculation
                    if "pos" in label.lower():
                        total_score += score
                        num_articles += 1
                    elif "neg" in label.lower():
                        total_score -= score
                        num_articles += 1

                elapsed_time = time.time() - start_time

        # 4. Overall sentiment summary
        if num_articles > 0:
            if total_score > 0.15:
                overall_sentiment = "Positive"
            elif total_score < -0.15:
                overall_sentiment = "Negative"
            else:
                overall_sentiment = "Neutral"

            st.markdown(f"""
            <div class="sentiment-box">
            <h4>Overall Sentiment: {overall_sentiment}</h4>
            <p>Aggregated Score: {total_score:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("No articles found matching the keyword.")

        # 5. Display performance stats
        st.write("---")
        st.write(f"**Model Selected:** {model_name}")
        st.write(f"**Inference Time (seconds):** {elapsed_time:.2f}")

if __name__ == "__main__":
    main()
