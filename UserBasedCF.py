#using Surprise library built-in dataset
#training on complete dataset and providing recommendations, no train-test split
from surprise import Dataset, KNNBasic
import pandas as pd

# Load dataset
data = Dataset.load_builtin('ml-100k')
full_trainset = data.build_full_trainset()  # Use full dataset for training

# Apply User-Based Collaborative Filtering using k-NN
sim_options = {
    'name': 'cosine',
    'user_based': True
}
algo = KNNBasic(sim_options=sim_options)
algo.fit(full_trainset)

all_movie_ids = list(full_trainset.all_items())
movie_id_map = {v: k for k, v in full_trainset._raw2inner_id_items.items()}  # Map back to original IDs

def recommend_movies(user_id, n_recommend=5):
    """ Recommend top N movies for a given user using User-Based Collaborative Filtering """
    
    user_id = str(user_id)  # Convert to string (Surprise requires this format)
    
    # Get list of movies already rated by the user
    rated_movies = {movie_id for (movie_id, _) in full_trainset.ur[full_trainset.to_inner_uid(user_id)]}

    # Predict ratings for all unseen movies
    predictions = [
        (movie_id_map[mid], algo.predict(user_id, movie_id_map[mid]).est)
        for mid in all_movie_ids if mid not in rated_movies
    ]

    # Sort by predicted rating in descending order
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Get top N recommended movies
    return predictions[:n_recommend]

# Recommend movies for a user
user_id = 196  # Example user
recommendations = recommend_movies(user_id, n_recommend=5)

# Convert to DataFrame and display
recommend_df = pd.DataFrame(recommendations, columns=['Movie ID', 'Predicted Rating'])
print(recommend_df)
