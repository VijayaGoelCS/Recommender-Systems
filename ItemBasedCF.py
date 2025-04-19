'''
This code is using item-based collaborative filtering to recommend movies based on 
nearest neighboring movies found using cosine similarity measure
(i.e., movies rated similarly by the same users).
'''
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
#from scipy.spatial.distance import pdist, squareform
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load datasets

df_selected = pd.read_excel("C:\\Users\\hp\\Desktop\\RatingFinal.xlsx", sheet_name=["users","movies", "ratings2000", "ratings2001"])

movies = df_selected["movies"]
ratings2000 = df_selected["ratings2000"]
ratings2001 = df_selected["ratings2001"]
users = df_selected["users"]
#print("Total unique users in ratings.xls:", ratings2000['UserId'].nunique())

# Create a user-item matrix (pivot table)
user_item_matrix = ratings2000.pivot(index="UserId", columns="MovieId", values="Rating").fillna(0)
#pd.reset_option("display.max_rows")
#pd.reset_option("display.max_columns")
#print(user_item_matrix.head())

# Compute user similarity using cosine similarity
user_similarity = cosine_similarity(user_item_matrix)

# Convert similarity matrix to a DataFrame for readability
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Display top 5 rows of user similarity
#print(user_similarity_df.head())

nn_algo = NearestNeighbors(metric='cosine')
nn_algo.fit(user_item_matrix.T)

class Recommender:
    def __init__(self):
        # This list will stored movies that called atleast ones using recommend_on_movie method
        self.hist = []
        self.ishist = False # Check if history is empty

    # This method will recommend movies based on a movie that passed as the parameter
    def recommend_on_movie(self,movie,n_reccomend = 3):
        self.ishist = True
       # movieid = int(movies[movies['Title']==movie, 'MovieId'].iloc[0])
        try:
          movieid = int(movies.loc[movies['Title'] == movie, 'MovieId'].iloc[0])
        except IndexError:
          raise ValueError(f"Movie '{movie}' not found in the dataset.")

        if movieid not in user_item_matrix.columns:
          raise ValueError(f"Movie ID {movieid} not found in user-item matrix.")

        self.hist.append(movieid)
      
        distance, neighbors = nn_algo.kneighbors([user_item_matrix.loc[:, movieid]], n_neighbors=n_reccomend+1)

# Retrieve correct Movie IDs from column indices
        movieids = [user_item_matrix.columns[i] for i in neighbors[0] ]
        print("Movie IDs Retrieved in neighborhood:", movieids)
        recommended_movies = movies[movies["MovieId"].isin(movieids)][["MovieId", "Genres"]]
        #recommended_movies.to_excel("C:\\Users\\hp\\Desktop\\recommended_movies.xlsx", index=False)
        file_path = "C:\\Users\\hp\\Desktop\\recommended_movies.xlsx"

# Append to Excel file
        if os.path.exists(file_path):
            with pd.ExcelWriter(file_path, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
               sheet = writer.sheets['Sheet1']
               start_row = sheet.max_row + 1  # Leave one row blank
               recommended_movies.to_excel(writer, index=False, header=False, startrow=start_row)
        else:
            recommended_movies.to_excel(file_path, index=False)


        print("Exported to recommended_movies.xlsx successfully!")
        recommends = [
                movies.loc[movies['MovieId'] == mid, 'Title'].iloc[0]
        if not movies.loc[movies['MovieId'] == mid, 'Title'].empty
        else 'Unknown'
        for mid in movieids if mid != movieid]

        return recommends[:n_reccomend]

    # This method will recommend movies based on history stored in self.hist list
    def recommend_on_history(self,n_recommend = 3):
        if self.ishist == False:
            return print('No history found')
        
        history = np.array([user_item_matrix.loc[:, mid] for mid in self.hist])
        print("Watched Movie IDs as History:", self.hist)
    # Compute average history vector
        avg_history = np.mean(history, axis=0).reshape(1, -1)

    # Find nearest neighbors
        distance, neighbors = nn_algo.kneighbors(avg_history, n_neighbors=n_recommend + len(self.hist))

    # Retrieve correct Movie IDs from column indices
        movieids = [user_item_matrix.columns[i] for i in neighbors[0]]
        print("recommended Movie IDs :", movieids)
    # Extract movie titles
        recommends = [
        movies.loc[movies['MovieId'] == mid, 'Title'].iloc[0]
        if not movies.loc[movies['MovieId'] == mid, 'Title'].empty else 'Unknown'
        for mid in movieids if mid not in self.hist
        ]

        return recommends[:n_recommend]
    
    
recommender = Recommender()
# Recommendation based on past watched movies, but the object just initialized. So, therefore no history found
recommender.recommend_on_history()

# Recommendation based on this movie
recommender.recommend_on_movie('Father of the Bride Part II (1995)')

# Recommendation based on past watched movies, and this time a movie is there in the history.
#recommender.recommend_on_history()

# Recommendation based on this movie
recommender.recommend_on_movie('Tigerland (2000)')

# Recommendation based on past watched movies, and this time two movies is there in the history.
#recommender.recommend_on_history()

# Recommendation based on this movie
recommender.recommend_on_movie('Dracula (1931)')

# Recommendation based on past watched movies, and this time three movies is there in the history.
#recommender.recommend_on_history()

# Recommendation based on this movie
recommender.recommend_on_movie('Money Train (1995)')

recommender.recommend_on_history()
