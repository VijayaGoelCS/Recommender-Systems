'''
This code finds similar users using cosine similarity
based on rating data of year 2020
'''
import pandas as pd
#import os
from sklearn.metrics.pairwise import cosine_similarity
#from scipy.spatial.distance import pdist, squareform
#import numpy as np
#from sklearn.neighbors import NearestNeighbors

# Load datasets

df_selected = pd.read_excel("C:\\Users\\hp\\Desktop\\RatingFinal.xlsx", sheet_name=["users","movies", "ratings2000", "ratings2001"])

movies = df_selected["movies"]
ratings2000 = df_selected["ratings2000"]
users = df_selected["users"]
#print("Total unique users in ratings.xls:", ratings2000['UserId'].nunique())

# Create a user-item matrix (pivot table)
user_item_matrix = ratings2000.pivot(index="UserId", columns="MovieId", values="Rating").fillna(0)

user_similarity = cosine_similarity(user_item_matrix)

# Convert similarity matrix to a DataFrame for readability
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Display top 5 rows of user similarity
#print(user_similarity_df.head())

# Function to find similar users
def get_similar_users(target_user, top_n=3):
    """Returns the top N most similar users to the target user."""
    if target_user not in user_similarity_df.index:
        return f"User {target_user} not found in dataset."
    similar_users = user_similarity_df[target_user].sort_values(ascending=False)[1:top_n+1]
    return similar_users

# Example: Get top 3 similar users for userId = 1
print(get_similar_users(1))