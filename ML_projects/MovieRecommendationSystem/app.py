import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from ast import literal_eval
from fuzzywuzzy import process

app = Flask(__name__)

# Load the data
credits_df = pd.read_csv("tmdb_5000_credits.csv")
movies_df = pd.read_csv("tmdb_5000_movies.csv")

# Preprocess the data
credits_df = credits_df.rename(columns={"movie_id": "id", "title": "credits_title", "cast": "credits_cast", "crew": "credits_crew"})
movies_df = movies_df.merge(credits_df, on="id")

# Weighted rating calculation
c = movies_df["vote_average"].mean()
m = movies_df["vote_count"].quantile(0.9)
new_movies_df = movies_df[movies_df["vote_count"] >= m].copy()

def weighted_rating(x, c=c, m=m):
    v = x["vote_count"]
    R = x["vote_average"]
    return (v / (v + m) * R) + (m / (v + m) * c)

new_movies_df["score"] = new_movies_df.apply(weighted_rating, axis=1)
new_movies_df = new_movies_df.sort_values("score", ascending=False)

# Content-based filtering setup
movies_df["overview"] = movies_df["overview"].fillna("")
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies_df["overview"])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()

# Metadata-based filtering setup
features = ["credits_cast", "credits_crew", "keywords", "genres"]
for feature in features:
    movies_df[feature] = movies_df[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]
        return names[:3] if len(names) > 3 else names
    return []

movies_df["director"] = movies_df["credits_crew"].apply(get_director)
for feature in ["credits_cast", "keywords", "genres"]:
    movies_df[feature] = movies_df[feature].apply(get_list)

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        return str.lower(x.replace(" ", "")) if isinstance(x, str) else ""

for feature in ["credits_cast", "keywords", "director", "genres"]:
    movies_df[feature] = movies_df[feature].apply(clean_data)

movies_df["soup"] = movies_df.apply(lambda x: " ".join(x["keywords"]) + " " + " ".join(x["credits_cast"]) + " " + x["director"] + " " + " ".join(x["genres"]), axis=1)

count_vectorizer = CountVectorizer(stop_words="english")
count_matrix = count_vectorizer.fit_transform(movies_df["soup"])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Helper functions
def get_closest_title(title, indices):
    titles = indices.index.to_list()
    closest_match = process.extractOne(title, titles)
    return closest_match[0] if closest_match else None

def get_recommendations(title, cosine_sim=cosine_sim):
    title = title.strip()
    if title not in indices:
        closest_title = get_closest_title(title, indices)
        return f"Did you mean: {closest_title}?" if closest_title else "Movie not found. Please try a different title."
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movies_indices = [i[0] for i in sim_scores]
    return movies_df["title"].iloc[movies_indices]

# Flask routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    movie_name = request.form["movie_name"]
    method = request.form["method"]

    if method == "content":
        recommendations = get_recommendations(movie_name)
    elif method == "metadata":
        recommendations = get_recommendations(movie_name, cosine_sim2)
    else:
        recommendations = "Invalid method selected."

    if isinstance(recommendations, str):
        return render_template("index.html", message=recommendations)

    recommendations_html = recommendations.to_frame(name="Recommendations").to_html(classes="table table-striped")
    return render_template("recommendations.html", tables=recommendations_html)

if __name__ == "__main__":
    app.run(debug=True)
