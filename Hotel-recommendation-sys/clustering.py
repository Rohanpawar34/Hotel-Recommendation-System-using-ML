from sklearn.cluster import KMeans
def cluster_hotels(df):
    clustered = df.copy()
    features = clustered[["Price", "Rating"]]

    model = KMeans(n_clusters=3,random_state=42,n_init=10)

    clustered["Cluster"] = ( model.fit_predict(features))
    return clustered