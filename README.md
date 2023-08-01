## Movie Recommender System 

This project implements a Content-Based Movie Recommender System with a Streamlit frontend and a Natural Language Processing (NLP) backend. The system recommends similar movies based on the content of the movie descriptions, genres, keywords, cast, and crew.

### Technical Skills Utilized

#### Frontend (Streamlit)
- Streamlit for building the user interface.
- Interfacing with a backend server to fetch movie recommendations.

#### Backend (Content-Based NLP)
- Python for scripting and data manipulation.
- Pandas for data loading and preprocessing.
- nltk library for natural language processing tasks.
- Pickle for serializing and deserializing Python objects.
- CountVectorizer from scikit-learn for text vectorization.
- Cosine similarity for measuring movie similarity.

### Dataset
To run this project, you need to download the dataset from Kaggle. You can find the dataset at the following link:
[https://www.kaggle.com/tmdb/tmdb-movie-metadata?select=tmdb_5000_movies.csv](https://www.kaggle.com/tmdb/tmdb-movie-metadata?select=tmdb_5000_movies.csv)

### Project Structure
```
├── frontend.py            # Streamlit frontend code
├── backend.py             # Content-Based NLP backend code
├── movie_dict.pkl         # Pickle file containing movie data dictionary
├── similarity.pkl         # Pickle file containing cosine similarity matrix
├── tmdb_5000_movies.csv   # Dataset with movie information (to be downloaded)
├── tmdb_5000_credits.csv  # Dataset with movie credits information (to be downloaded)
└── README.md              # Project documentation
```

### Instructions
1. Install the required dependencies by running `pip install -r requirements.txt`.
2. Download the datasets `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` from the Kaggle link provided above and place them in the project directory.
3. Execute the backend script (`backend.py`) to preprocess the data, vectorize movie tags using a Bag of Words model with CountVectorizer, and calculate movie similarities based on cosine distance.
4. The generated `movie_dict.pkl` and `similarity.pkl` files will be used by the frontend.
5. Run the frontend script (`frontend.py`) to launch the web-based Content-Based Movie Recommender System.
6. Select a movie from the dropdown menu and click the 'Recommend' button to view similar movie suggestions based on content.

Feel free to explore, modify, and enhance the code to suit your needs. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

Enjoy discovering great movies with the Content-Based Movie Recommender System!
