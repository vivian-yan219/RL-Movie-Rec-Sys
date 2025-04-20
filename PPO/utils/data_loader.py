import pandas as pd
import os

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess MovieLens 100k data.
    Returns (ratings_df, users_df, movies_df)

    ratings_df: user_id, movie_id, rating, timestamp
    users_df: user_id, gender, age, occupation, zip
    movies_df: movie_id, title, genres (one-hot genre columns)
    """
    
    ROOT_DIR = os.getcwd()
    ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..'))
    DATA_DIR = os.path.join(ROOT_DIR, 'ml-100k')

    ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))
    ratings.to_csv(os.path.join(DATA_DIR, 'mod_ratings.csv'), sep='\t', index=False)
    movies = pd.read_csv(os.path.join(DATA_DIR, 'movies.csv'))
    movies.to_csv(os.path.join(DATA_DIR, 'mod_movies.csv'), sep='\t', index=False)

    #Loading datasets
    ratings_list = [i.strip().split("\t") for i in open(os.path.join(DATA_DIR,'mod_ratings.csv'), 'r').readlines()]
    ratings_df = pd.DataFrame(ratings_list[1:], columns = ['user_id', 'movie_id', 'rating', 'timestamp'])
    ratings_df['user_id'] = ratings_df['user_id'].apply(pd.to_numeric)
    ratings_df['movie_id'] = ratings_df['movie_id'].apply(pd.to_numeric)
    ratings_df['rating'] = ratings_df['rating'].astype(float)

    movies_list = [i.strip().split("\t") for i in open(os.path.join(DATA_DIR,'mod_movies.csv'),encoding='latin-1').readlines()]
    movies_df = pd.DataFrame(movies_list[1:], columns = ['movie_id', 'title', 'genres'])
    movies_df['movie_id'] = movies_df['movie_id'].apply(pd.to_numeric)

    # Expand genres into one-hot
    genres_list = sorted({g for sub in movies_df['genres'].str.split('|') for g in sub})
    for genre in genres_list:
        movies_df[genre] = movies_df['genres'].str.contains(genre).astype(int)

    movies_df.drop(columns=['genres'], inplace=True)

    users_1m_path = os.path.join(DATA_DIR, 'users-1m.dat')
    users_1m_list = [i.strip().split("::") for i in open(users_1m_path, 'r').readlines()]
    users_1m_df = pd.DataFrame(users_1m_list, columns=['user_id', 'gender', 'age', 'occupation', 'zip_code'])
    users_1m_df['user_id'] = users_1m_df['user_id'].apply(pd.to_numeric)

    users_100k = users_1m_df[users_1m_df['user_id'].isin(ratings_df['user_id'])]
    users_100k.to_csv(os.path.join(DATA_DIR, 'users.csv'), sep='\t', index=False)

    users_list = [i.strip().split("\t") for i in open(os.path.join(DATA_DIR,'users.csv'), 'r').readlines()]
    users_df = pd.DataFrame(users_list[1:], columns=['user_id', 'gender', 'age', 'occupation', 'zip_code'])

    return ratings_df, users_df, movies_df