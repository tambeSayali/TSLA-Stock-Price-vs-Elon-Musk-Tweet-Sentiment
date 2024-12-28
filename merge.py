import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the datasets
tweets = pd.read_csv('sentiment_output_fixed.csv', parse_dates=['Timestamp'])
stocks = pd.read_csv('TSLA.csv', parse_dates=['Date'])

# Preprocess tweets
tweets['Date'] = tweets['Timestamp'].dt.date
tweets['Sentiment_Score'] = tweets['Sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})
daily_sentiments = tweets.groupby('Date')['Sentiment_Score'].mean().reset_index()

# Preprocess stock data
stocks['Date'] = stocks['Date'].dt.date
stocks['Daily_Change'] = stocks['Close'].pct_change()

# Merge datasets
merged = pd.merge(stocks, daily_sentiments, on='Date', how='inner')

# Analyze correlation
correlation = merged[['Daily_Change', 'Sentiment_Score']].corr()
print("Correlation Matrix:")
print(correlation)

# Clustering Analysis
features = merged[['Sentiment_Score', 'Daily_Change']].dropna()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
merged['Cluster'] = kmeans.fit_predict(scaled_features)

# Plot Clustering Results
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged, x='Sentiment_Score', y='Daily_Change', hue='Cluster', palette='viridis')
plt.title("Clustering Analysis: Sentiment vs. Daily Stock Price Change")
plt.xlabel("Sentiment Score")
plt.ylabel("Daily Stock Price Change (%)")
plt.legend(title="Cluster")
plt.show()

# Regression Analysis
X = merged[['Sentiment_Score']].dropna()
y = merged['Daily_Change'].dropna()
regressor = LinearRegression()
regressor.fit(X, y)

# Regression Results
plt.figure(figsize=(10, 6))
plt.scatter(merged['Sentiment_Score'], merged['Daily_Change'], label="Data Points", alpha=0.7)
plt.plot(X, regressor.predict(X), color='red', label='Regression Line')
plt.title("Regression Analysis: Sentiment vs. Daily Stock Price Change")
plt.xlabel("Sentiment Score")
plt.ylabel("Daily Stock Price Change (%)")
plt.legend()
plt.show()

# Volatility Analysis
merged['Volatility'] = merged['Close'].pct_change().rolling(window=7).std()  # 7-day rolling volatility
plt.figure(figsize=(12, 6))
plt.plot(merged['Date'], merged['Volatility'], label='Volatility (7-day rolling)', color='purple')
plt.title("Stock Volatility Over Time")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.show()

# Cumulative Sentiment and Returns
merged['Cumulative_Sentiment'] = merged['Sentiment_Score'].cumsum()
merged['Cumulative_Returns'] = (1 + merged['Daily_Change']).cumprod() - 1

plt.figure(figsize=(12, 6))
plt.plot(merged['Date'], merged['Cumulative_Sentiment'], label="Cumulative Sentiment", color='orange')
plt.plot(merged['Date'], merged['Cumulative_Returns'], label="Cumulative Returns", color='blue')
plt.title("Cumulative Sentiment and Stock Returns Over Time")
plt.xlabel("Date")
plt.ylabel("Cumulative Value")
plt.legend()
plt.show()

# Plot data per year
merged['Year'] = pd.to_datetime(merged['Date']).dt.year

for year in merged['Year'].unique():
    yearly_data = merged[merged['Year'] == year]
    plt.figure(figsize=(10, 6))
    plt.title(f'Tesla Stock Price vs Sentiment ({year})')
    plt.plot(yearly_data['Date'], yearly_data['Close'], label='Stock Price', color='blue')
    plt.plot(yearly_data['Date'], yearly_data['Sentiment_Score'] * 100, label='Sentiment Score (scaled)', color='orange')
    plt.legend()
    plt.show()

# Scatter plot for all years combined
plt.figure(figsize=(12, 8))
for year in merged['Year'].unique():
    yearly_data = merged[merged['Year'] == year]
    plt.scatter(yearly_data['Sentiment_Score'], yearly_data['Daily_Change'], label=f'{year}')
plt.title('Sentiment Score vs Daily Stock Price Change')
plt.xlabel('Sentiment Score')
plt.ylabel('Daily Stock Price Change (%)')
plt.legend()
plt.show()
