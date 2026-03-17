import yfinance as yf
import pandas as pd
from src.util.log_config import setup_logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
logger = setup_logging('yahoo')

def sentiment(ticker: yf.Ticker):
    news = ticker.news
    assert news 
    
    #Retrieve only the content  
    filtered_news = list(map(lambda x: x.get('content'), news))
    
    df = pd.DataFrame(filtered_news)
    assert 'title' in df.columns
    scores = df['title'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

    return {
        'mean': float(scores.mean()),
        'news': news,
        'price_targets': ticker.analyst_price_targets
        }

def trading_data(ticker: yf.Ticker):
    """Method to extract only the price data needed 
    for the Valuation agent 
    """
    day = ticker.history('1d')
    #Filtering for Date, Open Price and Volume 
    month = ticker.history('1mo')
    mdf = pd.DataFrame(month)
    mdf = mdf[['Open', 'Volume']].reset_index()
    year = ticker.history('1y')

    ydf = pd.DataFrame(year)
    ydf = ydf[['Open', 'Volume']].reset_index()
    
    # Daily price calculation, 5d ensures that latest full day is included 
    range = ticker.history('5d').iloc[-1]

    return {
        'price': {
            'Day': day,      #Price data for a day
            'Month': mdf,   # month
            'Year': ydf,     # year
            'High': range['High'],              # days high
            'Low' : range['Low'],               # days low 
            'Open': range['Open'],              # days open
            'Close': range['Close']             # days close 
        },
        'volume': {
            '1d' : day['Volume'],
            '1mo': mdf['Volume'],
            '1y': ydf['Volume']
        }
    }
    
def retrieve_yahoo_data(ticker: str):
    yfTicker = yf.Ticker(ticker)
    sentiment_data = sentiment(yfTicker)
    td = trading_data(yfTicker)
    price = td['price']
    volume = td['volume']
    data =  {
        'sentiment': {
            'mean': sentiment_data['mean'],
            'news': sentiment_data['news'],
            'price_targets': sentiment_data['price_targets']
        },
        'price': {
            'day': price['Day'],      #Price data for a day
            'month': price['Month'],   # month
            'year': price['Year'],     # year
            'high': price['High'],              # days high
            'low' : price['Low'],               # days low
            'open': price['Open'],              # days open
            'close': price['Close']             # days close
        },
        'volume': {
            '1d' : volume['1d'],
            '1mo': volume['1mo'],
            '1y': volume['1y']
        }
    }

    return data


def format_valuation_context(data: dict) -> str:
    """Format Yahoo price/volume data into a rich context string for the Valuation agent."""
    lines = []

    # Current day OHLC
    lines.append(f"Current Day Prices:")
    lines.append(f"  Open: ${data['price']['open']:.2f}")
    lines.append(f"  High: ${data['price']['high']:.2f}")
    lines.append(f"  Low:  ${data['price']['low']:.2f}")
    lines.append(f"  Close: ${data['price']['close']:.2f}")

    # 1-month trend
    month_df = data['price']['month']
    if month_df is not None and len(month_df) > 1:
        month_start = float(month_df['Open'].iloc[0])
        month_end = float(month_df['Open'].iloc[-1])
        month_pct = ((month_end - month_start) / month_start) * 100
        month_high = float(month_df['Open'].max())
        month_low = float(month_df['Open'].min())
        month_avg_vol = float(data['volume']['1mo'].mean())
        lines.append(f"\n1-Month Trend ({len(month_df)} trading days):")
        lines.append(f"  Start: ${month_start:.2f} → End: ${month_end:.2f} ({month_pct:+.1f}%)")
        lines.append(f"  Month High: ${month_high:.2f} | Month Low: ${month_low:.2f}")
        lines.append(f"  Avg Daily Volume: {month_avg_vol:,.0f} shares")

    # 1-year trend
    year_df = data['price']['year']
    if year_df is not None and len(year_df) > 1:
        year_start = float(year_df['Open'].iloc[0])
        year_end = float(year_df['Open'].iloc[-1])
        year_pct = ((year_end - year_start) / year_start) * 100
        year_high = float(year_df['Open'].max())
        year_low = float(year_df['Open'].min())
        year_avg_vol = float(data['volume']['1y'].mean())
        lines.append(f"\n1-Year Trend ({len(year_df)} trading days):")
        lines.append(f"  Start: ${year_start:.2f} → End: ${year_end:.2f} ({year_pct:+.1f}%)")
        lines.append(f"  52-Week High: ${year_high:.2f} | 52-Week Low: ${year_low:.2f}")
        lines.append(f"  Avg Daily Volume: {year_avg_vol:,.0f} shares")

    return "\n".join(lines)


def format_sentiment_context(data: dict) -> str:
    """Format Yahoo sentiment/news data into a rich context string for the Sentiment agent."""
    lines = []

    # Overall sentiment
    mean_sent = data['sentiment']['mean']
    if mean_sent > 0.2:
        label = "Bullish"
    elif mean_sent < -0.2:
        label = "Bearish"
    else:
        label = "Neutral"
    lines.append(f"Overall Sentiment Score: {mean_sent:.2f} (scale: -1.0 bearish to +1.0 bullish) — {label}")

    # News headlines with individual scores
    news = data['sentiment']['news']
    if news:
        lines.append(f"\nRecent News ({len(news)} articles):")
        for i, article in enumerate(news[:10]):
            content = article.get('content', {})
            title = content.get('title', 'N/A') if isinstance(content, dict) else str(content)
            score = analyzer.polarity_scores(title)['compound'] if title != 'N/A' else 0.0
            lines.append(f"  {i+1}. \"{title}\" (sentiment: {score:+.2f})")

    # Analyst price targets
    targets = data['sentiment']['price_targets']
    if targets is not None:
        try:
            lines.append(f"\nAnalyst Price Targets:")
            lines.append(f"  Low:    ${targets.get('low', 'N/A')}")
            lines.append(f"  Mean:   ${targets.get('mean', 'N/A')}")
            lines.append(f"  Median: ${targets.get('median', 'N/A')}")
            lines.append(f"  High:   ${targets.get('high', 'N/A')}")
            lines.append(f"  Current: ${targets.get('current', 'N/A')}")
            num_analysts = targets.get('numberOfAnalystOpinions', 'N/A')
            lines.append(f"  Number of Analysts: {num_analysts}")
        except (AttributeError, TypeError):
            lines.append(f"\nAnalyst Price Targets: Not available")
    else:
        lines.append(f"\nAnalyst Price Targets: Not available")

    return "\n".join(lines)


