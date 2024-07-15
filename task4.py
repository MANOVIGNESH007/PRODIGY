import nltk
nltk.download('movie_reviews')

from nltk.corpus import movie_reviews
import pandas as pd

# Load movie reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Convert to DataFrame
reviews_df = pd.DataFrame(documents, columns=['words', 'category'])

# Join words into a single string for each review
reviews_df['review'] = reviews_df['words'].apply(lambda x: ' '.join(x))

from textblob import TextBlob

def get_sentiment(review):
    analysis = TextBlob(review)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

reviews_df['sentiment'] = reviews_df['review'].apply(get_sentiment)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(review):
    score = analyzer.polarity_scores(review)
    if score['compound'] > 0.05:
        return 'positive'
    elif score['compound'] < -0.05:
        return 'negative'
    else:
        return 'neutral'

reviews_df['vader_sentiment'] = reviews_df['review'].apply(get_vader_sentiment)

import matplotlib.pyplot as plt
import seaborn as sns

# Count the occurrences of each sentiment
sentiment_counts = reviews_df['sentiment'].value_counts()
vader_sentiment_counts = reviews_df['vader_sentiment'].value_counts()

# Plot the sentiment distribution using TextBlob
plt.figure(figsize=(12, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.title('Sentiment Analysis of Movie Reviews (TextBlob)')
plt.savefig("Sentiment Analysis of Movie Reviews (TextBlob).png")
plt.show()

# Plot the sentiment distribution using VADER
plt.figure(figsize=(12, 6))
sns.barplot(x=vader_sentiment_counts.index, y=vader_sentiment_counts.values, palette='viridis')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.title('Sentiment Analysis of Movie Reviews (VADER)')
plt.savefig("Sentiment Analysis of Movie Reviews (VADER).png")
plt.show()
