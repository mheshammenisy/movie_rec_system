import polars as pl
import scipy.sparse as sp
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # Finds C:\movie_recommendation
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

def run_pipeline():
    # 1. Load Files
    movies_df = pl.read_parquet(DATA_DIR / "processed" / "movies_final.parquet")
    links_df = pl.read_csv(DATA_DIR / "raw" / "links.csv")
    
    movies_df = movies_df.join(links_df.select(["movieId", "tmdbId"]), on="movieId", how="left")
    movies_df = movies_df.drop_nulls(subset=["tmdbId"])

    # 3. Feature Engineering: Genres
    unique_genres = [g for g in movies_df.select("genres").with_columns(pl.col("genres").str.split("|")).explode("genres").unique().sort("genres")["genres"].to_list() if g != "(no genres listed)"]
    for genre in unique_genres:
        col_name = f"g_{genre.lower().replace(' ', '_').replace('-', '_')}"
        movies_df = movies_df.with_columns(pl.col("genres").str.contains(genre).cast(pl.Int8).alias(col_name))

    # 4. Feature Engineering: Year Scaling
    min_y, max_y = movies_df.filter(pl.col("year") > 0)["year"].min(), movies_df["year"].max()
    movies_df = movies_df.with_columns(((pl.col("year") - min_y) / (max_y - min_y)).clip(0, 1).alias("year_scaled"))

    # 5. Feature Engineering: Tags (TF-IDF)
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf.fit_transform(movies_df["tag"].to_list())

    # 6. Build Master Matrix
    feature_cols = [c for c in movies_df.columns if c.startswith("g_") or c == "year_scaled"]
    genre_year_matrix = sp.csr_matrix(movies_df.select(feature_cols).to_numpy())
    final_features = sp.hstack([genre_year_matrix, tfidf_matrix])

    # 7. SAVE EVERYTHING
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir()
    
    joblib.dump(final_features, MODEL_DIR / "final_features.joblib")
    movies_df.write_parquet(DATA_DIR / "processed" / "movies_featurized.parquet")
    
    print("✅ Pipeline Complete. Brain saved to /models/")

if __name__ == "__main__":
    run_pipeline()