# TSLA-Stock-Price-vs-Elon-Musk-Tweet-Sentiment

### **Objective of the Project:**
 Investigate how Elon Musk's Twitter (X) activity correlates with fluctuations in Tesla's stock price.

**Key Focus Areas:**

- Analyze the sentiment of tweets.
- Correlate tweet content with stock price movements.
- Identify specific events or tweets that significantly impact stock performance.

**Research Questions**

1. How does the sentiment of Elon Musk's tweets affect Tesla's stock price?
2. Are there specific tweets that correlate with significant stock price movements?
3. What patterns exist in the timing of tweets and stock volatility?
4. How do different types of tweets impact stock price differently?

### ****Data Collection****

**Tweet Datasets**

- June 2010 - March 2022: [elonmusk_elon_tweets (HuggingFace)](https://huggingface.co/datasets/MasaFoundation/elonmusk_elon_tweets)
- January 2022 - October 2022: [Elon Musk’s Tweet Dataset 2022 (Kaggle)](https://www.kaggle.com/datasets/marta99/elon-musks-tweets-dataset-2022?resource=download)
- July 2022 - June 2023: [Collect Elon Musk Tweets (Kaggle)](https://www.kaggle.com/code/gpreda/collect-elon-musk-tweets/output)

**TSLA Stock Price Dataset**

- [Yahoo Finance TSLA Stock Price Data](https://finance.yahoo.com/quote/TSLA/?guccounter=1)**Merging Datasets and Sentiment Analysis**

### **Data Merging and Sentiment Analysis**

**Steps to Merge Tweet Datasets:**

1. Remove extraneous columns not present in all three datasets.
2. Merge datasets into a single file.
3. Sort by date and remove duplicate tweets.
4. Unify date column formatting.
5. Add a column for sentiment analysis scores.

**Sentiment Analysis**

- Tool: [RoBERTa Sentiment Analysis Model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) tuned specifically for tweets.
- Tweets were categorized as Positive, Negative, or Neutral based on their sentiment.

### **Analysis: Correlation Between Tweets and Stock Price**

 **Yearly Trends:**
- Visualized yearly correlations between tweet sentiment and TSLA stock price movements.
- Identified periods of increased tweet activity and stock volatility.

 **Sentiment Impact:**
- Assessed stock price changes following Positive, Negative, and Neutral tweets.
- Highlighted trends in tweet sentiment over time.

 **Timing Patterns:**
- Analyzed stock reactions to tweets during trading hours vs. after hours.
- Compared short-term vs. long-term stock price effects.

 **Significant Events:**
- Identified key tweets causing major stock price changes.
- Studied stock behavior before and after high-impact tweets.

 **Volatility Analysis:**
- Examined stock price fluctuations relative to tweet frequency and sentiment.
- Cross-referenced volatility with external market factors.

**Tweet Types:**
- Compared Tesla-related tweets with general or personal tweets.
- Measured their influence on stock performance.

 **Advanced Correlations:**
- Combined sentiment analysis with market indicators like trading volume and moving averages.

### **Conclusion**
This project highlights the potential impact of Elon Musk's Twitter activity on Tesla's stock price, showcasing correlations between tweet sentiment, timing, and market volatility. By combining sentiment analysis with stock data, the study provides insights into how social media influences financial markets.

### **Future Work**

- Expand analysis to include tweets from other influential figures and their impact on related stocks.
- Incorporate additional stock market metrics for deeper insights.
- Enhance sentiment analysis by fine-tuning models for better accuracy on tweet-specific language.
- Automate data collection and analysis for real-time tracking of social media's effect on stock prices.
