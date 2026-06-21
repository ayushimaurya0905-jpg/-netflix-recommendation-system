# -netflix-recommendation-system
A recommendation system built on the Netflix Prize dataset using SVD and Item-Based Collaborative Filtering
# Netflix Recommendation System

A personalized movie recommendation system built on the Netflix Prize dataset, 
implementing and comparing two recommendation approaches: SVD (Matrix Factorization) 
and Item-Based Collaborative Filtering.

## Project Overview

This project was built as a self-directed learning exercise to understand how 
recommendation systems work — the same kind of system that powers Netflix, 
Amazon, and Spotify. It uses real-world data from the historic Netflix Prize 
competition (2006).

## Dataset

- **Source:** [Netflix Prize Dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)
- **Subset used:** First 500,000 ratings (full dataset has 100M+ ratings — 
  too large for free compute)
- **Format:** UserID, MovieID, Rating (1-5 stars), Date

## Approach

### Model 1: SVD (Matrix Factorization)
Implemented from scratch in NumPy using the FunkSVD algorithm (the same method 
Simon Funk originally used for this exact competition). Learns hidden "taste 
factors" for each user and movie through gradient descent.

*Note: built manually instead of using the `scikit-surprise` library, which has 
compatibility issues with current NumPy versions on Kaggle.*

### Model 2: Item-Based Collaborative Filtering
Built using cosine similarity between movies (rather than users) based on 
shared rating patterns. Chosen over user-based CF for computational efficiency, 
since the number of movies is much smaller than the number of users.

## Evaluation Methodology

- **Train/test split:** 80% training, 20% testing (random split, seed=42)
- **Relevance threshold:** A movie is considered "relevant" to a user if their 
  actual rating is ≥ 3.5 stars
- **MAP@10 methodology:** For each user, their held-out test items were ranked 
  by predicted rating, and Average Precision@10 was computed against actual 
  relevant items, then averaged across all users

## Results

| Model                          | RMSE      | MAP@10    |
|--------------------------------|-----------|-----------|
| SVD (Matrix Factorization)     | 0.9954    | 0.9691    |
| Item-Based Collaborative Filtering | 1.1759| 0.9570    |

## Key Insights

- SVD performed better because it learns hidden patterns in user behavior, while Item-Based Collaborative Filtering only uses direct similarity between movies.
- Data is highly sparse — even the most active users have rated only a small 
  fraction of available movies
- sparsity % is 44.0%

## Sample Recommendations

**Success case:** User 667426 received highly relevant recommendations 
(AP@10 = 1.00 (1 relevant out of 1 test movies))

**Failure case:** User 1932594 received less accurate recommendations 
(AP@10 = 0.00), likely due to  limited training history

## How to Run

1. Open the dataset on Kaggle: [Netflix Prize Data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)
2. Create a new notebook with this dataset attached
3. Upload `notebook6f875c8c41.ipynb` or copy its cells into a new notebook
4. Run all cells in order

## What I Learned

This was my first hands-on machine learning project. Key concepts I learned 
include collaborative filtering, matrix factorization, train/test splitting, 
and evaluating recommendation quality using RMSE and MAP@10.

## Tech Stack

Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, SciPy
