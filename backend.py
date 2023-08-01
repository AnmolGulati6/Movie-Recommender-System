import numpy as np
import pandas as pd
import nltk
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies.head(1)  # shows first row with headers
credits.head(1)  # shows first row with headers

movies = movies.merge(credits, on='title')

# will keep the following categories: genres, id, keywords, title, overview, cast, crew

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

movies.isnull().sum()
movies.dropna(inplace=True)  # removed 3 movies that didn't have overviews
movies.duplicated().sum()  # no row is duplicated, all unique
print(movies.iloc[0].genres)  # prints in a weird format


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):  # convert string to list
        L.append(i['name'])
    return L


def top_3_Cast(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):  # convert string to list
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L


def fetch_Director(obj):
    L = []
    for i in ast.literal_eval(obj):  # convert string to list
        if i['job'] == 'Director':
            L.append(i['name'])
    return L


ps = PorterStemmer()


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)



movies['genres'] = movies['genres'].apply(convert)  # will store all genres of every movie in a list
movies['keywords'] = movies['keywords'].apply(convert)  # will store all keywords of every movie in a list
movies['cast'] = movies['cast'].apply(top_3_Cast)  # will store only top 3 actors of every movie in a list
movies['crew'] = movies['crew'].apply(fetch_Director)  # will store only director of every movie in a list
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
new_df['tags'] = new_df['tags'].apply(stem)

# Using Bag of Words model
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()  # every movie is in a vector form

# print(cv.get_feature_names_out()) all strings are stemmed


# Calculate distance between each movie using cosine distance, eucladian distance not good measure
similarity = cosine_similarity(vectors)


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x: x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))