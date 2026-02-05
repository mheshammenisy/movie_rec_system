# 🎬 AI-Powered Movie Recommender

A content-based recommendation engine built with **Polars**, **Scikit-Learn**, and **Streamlit**. 
This system uses TF-IDF vectorization and Cosine Similarity to find movies based on genres, years, and 30M+ user tags.

## 🚀 How to Run
Due to GitHub's file size limits, the raw dataset is not included in this repository. 

### 1. Setup Data
* Download the **MovieLens 25M Dataset** from [grouplens.org](https://grouplens.org/datasets/movielens/25m/).
* Extract the ZIP and place the CSV files in `data/raw/`.

### 2. Install Requirements
```bash
pip install -r requirements.txt