import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from review_sentiment import process_reviews

df = pd.read_csv("hotels.csv")
df.columns = df.columns.str.strip()
# ======================================
# Clean Price Column 
# ======================================
def extract_price(value):
    import re
    nums = re.findall(r"\d+", str(value))
    return float(nums[0]) if nums else np.nan

df["price"] = df["Price"].apply(extract_price)

df["price"] = df["price"].fillna(df["price"].median())

# keep compatibility
df["Price"] = df["price"]

# ======================================
# FEATURES
# ======================================

df["features"] = (df["City"].astype(str) + " " + df["Amenities"].astype(str) + " " + df["Rating"].astype(str))
vectorizer = TfidfVectorizer()

feature_matrix = vectorizer.fit_transform(df["features"])

similarity = cosine_similarity(feature_matrix)
# ======================================
# SENTIMENT
# ======================================
try:
    sentiment_df = process_reviews()
    df = df.merge(sentiment_df,on="Hotel_Name",how="left")

    df["sentiment_score"] = (df["sentiment_score"].fillna(0))

except:
    df["sentiment_score"] = 0
# ======================================
# RECOMMEND HOTELS
# ======================================
def recommend_hotels(city,budget,rating,amenities):

    result = df.copy()

    result = result[(result["City"] == city)&(result["Price"] <= budget)&(result["Rating"] >= rating)]

    for amenity in amenities:
        result = result[result["Amenities"].str.contains(amenity,case=False,na=False)]

    if result.empty:
        return result

    result["final_score"] = (0.6 * result["Rating"]+0.4 * result["sentiment_score"])
    result = result.sort_values(by="final_score",ascending=False)

    return result
# ======================================
# SIMILAR HOTELS
# ======================================
def similar_hotels(hotel_name):

    if hotel_name not in df["Hotel_Name"].values:
        return pd.DataFrame()

    idx = df[df["Hotel_Name"] == hotel_name].index[0]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommendations = []

    for i in scores[1:6]:
        recommendations.append(df.iloc[i[0]])

    return pd.DataFrame(recommendations)