import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

columns = ['id', 'topic', 'sentiment_label', 'text']

train_path = "dataset/twitter_training.csv"
val_path = "dataset/twitter_validation.csv"

train_df = pd.read_csv(train_path, names=columns, header=None)
val_df = pd.read_csv(val_path, names=columns, header=None)

def clean_text(text):
    text = re.sub(r'http\S+', '', str(text))  # Remove URLs
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Remove mentions
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase and trim spaces
    return text

train_df['clean_text'] = train_df['text'].apply(clean_text)
val_df['clean_text'] = val_df['text'].apply(clean_text)

sia = SentimentIntensityAnalyzer()

train_df['sentiment_score'] = train_df['clean_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
val_df['sentiment_score'] = val_df['clean_text'].apply(lambda x: sia.polarity_scores(x)['compound'])

def classify_sentiment(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

train_df['sentiment'] = train_df['sentiment_score'].apply(classify_sentiment)
val_df['sentiment'] = val_df['sentiment_score'].apply(classify_sentiment)

plt.figure(figsize=(8, 5))
sns.countplot(x='sentiment', data=train_df, palette='coolwarm')
plt.title('Sentiment Distribution in Training Data')
plt.show()

for sentiment in ['Positive', 'Negative']:
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(
        ' '.join(train_df[train_df['sentiment'] == sentiment]['clean_text'])
    )
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment} Sentiments')
    plt.show()

print(train_df[['text', 'sentiment']].head())
