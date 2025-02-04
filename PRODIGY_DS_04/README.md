# Task Description
This task involves analyzing and visualizing sentiment patterns in social media data to understand public opinion and attitudes towards specific topics or brands. The goal is to:

 •	Extract sentiment from text data using NLP techniques.

 •	Classify sentiments as positive, negative, or neutral.

 •	Visualize trends to gain insights into public perception.

# Dataset

 •	Dataset Name: Twitter Sentiment Analysis
	
 •	Link: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
	
 •	Target Variable: Sentiment (Positive, Negative, Neutral)

# Requirements
Install necessary Python libraries:

    pip install pandas numpy seaborn matplotlib nltk wordcloud scikit-learn textblob

# Steps Performed
	
 1.	Data Preprocessing
	
 •	Cleaned text (removed stopwords, special characters, and links).
	
 •	Tokenized and normalized text data.

 2.	Sentiment Analysis
	
 •	Used TextBlob/VADER for sentiment classification.

 •	Assigned polarity scores to classify sentiments.
	
 3.	Data Visualization
	
 •	Bar chart for sentiment distribution.
	
 •	Word cloud for frequent words in positive and negative sentiments.
	
 •	Time series plots to analyze sentiment trends over time.
