import streamlit as st
import polars as pl
import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")



st.set_page_config(page_title="Movie Matcher", layout="wide")
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

@st.cache_resource
def load_assets():
    movie_path = DATA_DIR / "processed" / "movies_featurized.parquet"
    model_path = MODEL_DIR / "final_features.joblib"
    
    df = pl.read_parquet(movie_path)
    matrix = joblib.load(model_path)
    return df, matrix

movies_df, final_features = load_assets()

def get_poster(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}"
    try:
        data = requests.get(url).json()
        return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster"

def get_recommendations(target_title, n=5):
    idx = movies_df.select(pl.arg_where(pl.col("display_title") == target_title))[0,0]
    scores = cosine_similarity(final_features[idx], final_features).flatten()
    indices = scores.argsort()[-(n+1):-1][::-1]
    return movies_df[indices, :].with_columns(pl.Series("score", scores[indices]))

# --- UI ---
st.title("🎬 AI Movie Recommender")
choice = st.selectbox("Search for a movie:", movies_df["display_title"].to_list())

if st.button("Recommend"):
    recs = get_recommendations(choice)
    cols = st.columns(5)
    for i, row in enumerate(recs.to_dicts()):
        with cols[i]:
            st.image(get_poster(row['tmdbId']))
            st.write(f"**{row['display_title']}**")
            st.caption(f"Match: {int(row['score']*100)}%")