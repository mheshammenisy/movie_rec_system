# 🎬 AI-Powered Movie Recommender

A content-based movie recommendation system built using Polars, Scikit-Learn, and Streamlit.
The application recommends similar movies based on genres, release year, and over 30 million user-generated tags from the MovieLens dataset.

# 🌐 Live Demo

👉 Streamlit App:
https://movierecsystem-grsuppsuulqqsb25p9zt4x.streamlit.app/

# 🧠 Project Overview

This project implements a Content-Based Filtering recommendation engine using Natural Language Processing techniques.

# Core Features

TF-IDF vectorization of movie metadata and tags

Cosine similarity matching

Real-time movie recommendations

Poster fetching via TMDB API

Interactive Streamlit interface

Optimized data processing using Polars

# ⚙️ Tech Stack

Python

Streamlit

Polars

Scikit-Learn

Joblib

TMDB API

# 📊 Dataset

This project uses the MovieLens 25M dataset provided by GroupLens.

Due to GitHub file size limitations, raw datasets are not included in this repository.

# 🚀 How to Run Locally
1️⃣ Setup Data

Download the dataset from:

👉 https://grouplens.org/datasets/movielens/25m/

Extract the ZIP file and place the CSV files inside:

data/raw/

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Data Processing Notebook

Run the notebook to generate processed data and feature matrices.

4️⃣ Run the Streamlit App
streamlit run src/app.py

# 🔑 API Configuration

This project uses TMDB to fetch movie posters.

Create a TMDB API key from:

👉 https://www.themoviedb.org/settings/api

For local development, store your key in a .env file:

TMDB_API_KEY=your_api_key_here


For Streamlit Cloud deployment, add the key in Streamlit Secrets.

# 📌 Recommendation Methodology

Combine movie metadata and tags

Apply TF-IDF vectorization

Compute cosine similarity

Return top-N most similar movies

# 📷 Example Output

Users select a movie and receive visually enriched recommendations with similarity scores and posters.

# 🏗 Future Improvements

Hybrid recommendation (content + collaborative filtering)

User personalization

Model optimization with embeddings (BERT / Word2Vec)

Advanced filtering options

Docker deployment

# 👨‍💻 Author

Mohamed Hesham
Sustainable Energy & Data Analytics Enthusiast
