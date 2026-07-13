import pandas as pd
from textblob import TextBlob

def calculate_sentiment(review):
  return TextBlob(str(review)).sentiment.polarity

def process_reviews():
    reviews = pd.read_csv( "reviews.csv")

    reviews["sentiment_score"] = (reviews["Review"].apply(calculate_sentiment))

    sentiment_scores = (reviews.groupby( "Hotel_Name")["sentiment_score"].mean().reset_index())
    return sentiment_scores