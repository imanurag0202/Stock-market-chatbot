import yfinance as yf
import numpy as np
import pandas as pd
import feedparser
from prophet import Prophet
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline


def fetch_data(ticker, period="6mo", interval="1d"):
    stock = yf.Ticker(ticker)
    try:
        df = stock.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}.")
    except Exception as e:
        raise ValueError(f"Failed to fetch data for {ticker}: {e}")
    df['Return'] = df['Close'].pct_change()
    return df, stock


def calculate_moving_average(df, window=20):
    """
    Calculates the simple moving average (SMA) of the closing price over a specified window.
    """
    return df['Close'].rolling(window=window).mean()


def calculate_ema(df, span=20):
    """
    Calculates the Exponential Moving Average (EMA) of the closing price over a specified span.
    """
    return df['Close'].ewm(span=span, adjust=False).mean()


def calculate_rsi(df, period=14):
    """
    Calculates the Relative Strength Index (RSI) of the stock's closing price over a specified period.
    """
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(df, window=20):
    """
    Calculates the Bollinger Bands (upper, middle, lower) based on the closing price over a specified window.
    """
    ma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    upper_band = ma + (2 * std)
    lower_band = ma - (2 * std)
    return ma, upper_band, lower_band


def calculate_beta(stock_returns, market_returns):
    """
    Calculates the beta value of a stock, comparing it with market returns.
    """
    covariance = np.cov(stock_returns[1:], market_returns[1:])[0][1]
    variance = np.var(market_returns[1:])
    return covariance / variance


def calculate_roi(ticker, period="1y"):
    df, stock = fetch_data(ticker, period)
    if df.empty:
        raise ValueError(f"No data available for ticker {ticker}")
    start_price = df['Close'].iloc[0]
    end_price = df['Close'].iloc[-1]
    return ((end_price - start_price) / start_price) * 100


def fetch_google_news_rss(query, max_entries=10):
    """
    Fetches the latest stock-related news headlines based on the query.
    """
    url = f"https://news.google.com/rss/search?q={query}+stock"
    try:
        feed = feedparser.parse(url)
        if feed.entries:
            headlines = [entry['title'] for entry in feed.entries[:max_entries]]
        else:
            headlines = []
    except Exception as e:
        headlines = []
        print(f"Error fetching news: {e}")
    return headlines


tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def finbert_to_score(label, confidence):
    """
    Converts the sentiment label and confidence score into a numeric sentiment score.
    """
    if label == "Positive":
        return confidence
    elif label == "Negative":
        return -confidence
    return 0.0


def analyze_news_sentiment_finbert(query):
    """
    Analyzes sentiment of the latest news headlines based on the query using FinBERT.
    """
    headlines = fetch_google_news_rss(query)
    if not headlines:
        return pd.DataFrame(columns=["Headline", "Sentiment_Label", "Confidence", "Sentiment_Score"])

    data = []
    for headline in headlines:
        result = finbert(headline)[0]
        label = result['label']
        confidence = result['score']
        score = finbert_to_score(label, confidence)
        data.append({
            "Headline": headline,
            "Sentiment_Label": label,
            "Confidence": confidence,
            "Sentiment_Score": score
        })
    return pd.DataFrame(data)


def fetch_fundamentals(stock):
    info = stock.info
    fundamentals = {
        "P/E Ratio": info.get('trailingPE', 'N/A'),
        "Forward P/E": info.get('forwardPE', 'N/A'),
        "Dividend Yield": info.get('dividendYield', 'N/A')
    }
    return fundamentals




# def forecast_with_prophet(df, forecast_days=30, plot=False):
#     from prophet import Prophet
#     import matplotlib.pyplot as plt

#     prophet_df = df[['Close']].reset_index()
#     prophet_df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

#     # 🛠️ Remove timezone if present
#     if prophet_df['ds'].dt.tz is not None:
#         prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)

#     model = Prophet(daily_seasonality=True)
#     model.fit(prophet_df)

#     future = model.make_future_dataframe(periods=forecast_days)
#     forecast = model.predict(future)

#     if plot:
#         fig = model.plot(forecast)
#         plt.title('Prophet Stock Price Forecast')
#         plt.xlabel('Date')
#         plt.ylabel('Price')
#         plt.show()

#     return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], model


# from statsmodels.tsa.arima.model import ARIMA
# import matplotlib.pyplot as plt

# def forecast_with_arima(df, forecast_days=30, order=(5, 1, 0), plot=False):
#     """
#     ARIMA-based stock price forecasting.
#     """
#     close_prices = df['Close']
    
#     # Ensure index is datetime without timezone
#     close_prices.index = pd.to_datetime(close_prices.index).tz_localize(None)
    
#     # Set a proper daily frequency
#     close_prices = close_prices.asfreq('D')

#     # Fill missing dates if necessary
#     close_prices = close_prices.fillna(method='ffill')

#     # Fit ARIMA model
#     model = ARIMA(close_prices, order=order)
#     model_fit = model.fit()

#     # Forecast
#     forecast = model_fit.forecast(steps=forecast_days)

#     if plot:
#         plt.figure(figsize=(12,6))
#         plt.plot(close_prices, label='Historical')
#         plt.plot(forecast.index, forecast, label='Forecast', color='red')
#         plt.title('ARIMA Stock Price Forecast')
#         plt.xlabel('Date')
#         plt.ylabel('Close Price')
#         plt.legend()
#         plt.grid(True)
#         plt.show()

#     return forecast, model_fit





if __name__ == "__main__":
    # Example usage
    stock_symbol = "AAPL"
    df, stock = fetch_data(stock_symbol)

# forecast, model_fit = forecast_with_arima(df, forecast_days=30, plot=True)
# forecast, model = forecast_with_prophet(df, forecast_days=30, plot=True)
