# Stock Sentiment Analysis App

This is a web application that analyzes the sentiment of news articles related to a specific stock. The app uses a pre-trained model from Hugging Face's Transformers library to perform sentiment analysis on news articles fetched from Yahoo Finance.

## Features

- Input a stock ticker and a keyword to filter relevant news articles.
- Fetches news articles from Yahoo Finance RSS feeds.
- Analyzes the sentiment of each article using a financial sentiment analysis model.
- Displays the sentiment and score for each article.
- Provides an overall sentiment score for the stock based on the analyzed articles.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Sentinmental-analysis.git
   cd Sentinmental-analysis
   ```

2. **Install the required packages:**

   Make sure you have Python installed. Then, install the dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application:**

   Start the Streamlit app by running:

   ```bash
   streamlit run main.py
   ```

2. **Interact with the app:**

   - Enter a stock ticker (e.g., META) in the input field.
   - Optionally, enter a keyword to filter articles.
   - Click the "Analyze" button to fetch and analyze articles.

3. **View results:**

   - The app will display each article's title, link, publication date, summary, and sentiment.
   - An overall sentiment score for the stock will be shown at the bottom.

## Code Overview

- **`main.py`:** The main application file that sets up the Streamlit interface and performs sentiment analysis.
  - Imports necessary libraries and sets up the sentiment analysis pipeline.
  - Fetches and filters news articles based on user input.
  - Analyzes and displays sentiment for each article and calculates an overall sentiment score.

- **`requirements.txt`:** Lists all the Python packages required to run the application.

## Dependencies

The application relies on several Python packages, including but not limited to:

- `streamlit`
- `feedparser`
- `transformers`
- `requests`

For a full list of dependencies, see the `requirements.txt` file.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- The sentiment analysis model is sourced from [Hugging Face](https://huggingface.co/).
- News articles are fetched from [Yahoo Finance](https://finance.yahoo.com/).
