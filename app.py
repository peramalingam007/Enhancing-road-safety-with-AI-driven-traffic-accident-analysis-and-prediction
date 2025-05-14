import streamlit as st
import pandas as pd
import requests
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process, fuzz
import json

# --- Configuration ---
try:
    API_KEY = st.secrets["API_KEY"]  # Load from .streamlit/secrets.toml
except KeyError:
    API_KEY = os.getenv("TMDB_API_KEY")  # Fallback to environment variable
    if not API_KEY:
        st.error("API key not found. Please set it in Streamlit secrets or environment variables.")
        st.stop()

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    """Load and preprocess the movie dataset."""
    if not os.path.exists("merged_movies.csv"):
        st.error("Dataset 'merged_movies.csv' not found!")
        st.stop()
    
    df = pd.read_csv("merged_movies.csv")
    df.columns = df.columns.str.strip()
    required_columns = ["title", "overview", "popularity", "genres"]
    if not all(col in df.columns for col in required_columns):
        st.error("Dataset missing required columns: title, overview, popularity, genres")
        st.stop()
    
    df["overview"] = df["overview"].fillna("Overview not available")
    df["genres"] = df["genres"].fillna("")  # Ensure genres column is not null
    return df

# --- TF-IDF and Similarity Computation ---
@st.cache_resource
def compute_similarity(df):
    """Compute TF-IDF vectors and cosine similarity matrix for overviews."""
    tfidf = TfidfVectorizer(stop_words="english")
    vector = tfidf.fit_transform(df["overview"])
    similarity = cosine_similarity(vector)
    return similarity

# --- TMDb API Functions ---
@st.cache_data
def get_movie_info(movie_title):
    """Fetch movie details from TMDb API."""
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data["results"]:
            result = data["results"][0]
            movie_id = result["id"]
            # Fetch additional details for crew
            details_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}"
            details_response = requests.get(details_url)
            details_response.raise_for_status()
            credits = details_response.json()
            director = next((crew["name"] for crew in credits.get("crew", []) if crew["job"] == "Director"), "N/A")
            cast = [actor["name"] for actor in credits.get("cast", [])[:3]]  # Top 3 actors
            return {
                "poster_url": f"https://image.tmdb.org/t/p/w500{result['poster_path']}" if result.get("poster_path") else None,
                "rating": result.get("vote_average", "N/A"),
                "vote_count": result.get("vote_count", "N/A"),
                "genres": result.get("genre_ids", []),
                "director": director,
                "cast": cast,
                "movie_id": movie_id
            }
        return {"poster_url": None, "rating": "N/A", "vote_count": "N/A", "genres": [], "director": "N/A", "cast": [], "movie_id": None}
    except requests.RequestException as e:
        st.error(f"Error fetching movie info for '{movie_title}': {e}")
        return {"poster_url": None, "rating": "N/A", "vote_count": "N/A", "genres": [], "director": "N/A", "cast": [], "movie_id": None}

@st.cache_data
def get_top_rated_by_genre(genre_id):
    """Fetch top-rated movies for a specific genre from TMDb."""
    try:
        url = f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&sort_by=vote_average.desc&with_genres={genre_id}&vote_count.gte=100"
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get("results", [])[:5]
    except requests.RequestException as e:
        st.error(f"Error fetching top-rated movies for genre ID {genre_id}: {e}")
        return []

@st.cache_data
def get_movies_by_crew(crew_name, exclude_movie_id):
    """Fetch movies by a specific crew member (e.g., director, actor)."""
    try:
        # Search for person
        url = f"https://api.themoviedb.org/3/search/person?api_key={API_KEY}&query={crew_name}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data["results"]:
            person_id = data["results"][0]["id"]
            # Get movies associated with this person
            credits_url = f"https://api.themoviedb.org/3/person/{person_id}/movie_credits?api_key={API_KEY}"
            credits_response = requests.get(credits_url)
            credits_response.raise_for_status()
            movies = credits_response.json().get("cast", []) + credits_response.json().get("crew", [])
            # Filter out the searched movie and limit to 5
            return [movie for movie in movies if movie["id"] != exclude_movie_id][:5]
        return []
    except requests.RequestException as e:
        st.error(f"Error fetching movies for crew '{crew_name}': {e}")
        return []

# --- Recommendation Logic ---
def recommend(movie_title, df, similarity):
    """Generate recommendations based on genre, crew, and top-rated genre movies."""
    movie_title_lower = movie_title.lower()
    movie_list = df["title"].str.lower().tolist()
    results = {"searched_movie": None, "same_genre": [], "same_crew": [], "top_rated_genre": []}

    # Fuzzy matching for search
    match = process.extractOne(movie_title_lower, movie_list, scorer=fuzz.token_sort_ratio)
    if match and match[1] > 80:
        idx = movie_list.index(match[0])
        searched_movie_title = df.loc[idx, "title"]
        searched_movie_overview = df.loc[idx, "overview"]
        searched_movie_genres = df.loc[idx, "genres"].split(",") if df.loc[idx, "genres"] else []
        movie_info = get_movie_info(searched_movie_title)

        # Store searched movie details
        results["searched_movie"] = {
            "title": searched_movie_title,
            "overview": searched_movie_overview,
            "genres": searched_movie_genres,
            "movie_info": movie_info
        }

        # Same genre recommendations (using dataset)
        if searched_movie_genres:
            genre_matches = df[df["genres"].str.contains("|".join(searched_movie_genres), case=False, na=False)]
            genre_matches = genre_matches[genre_matches["title"] != searched_movie_title].head(5)
            results["same_genre"] = [
                {"title": row["title"], "overview": row["overview"]}
                for _, row in genre_matches.iterrows()
            ]

        # Same crew recommendations (using TMDb API)
        if movie_info["director"] != "N/A":
            crew_movies = get_movies_by_crew(movie_info["director"], movie_info["movie_id"])
            results["same_crew"] = [
                {"title": movie["title"], "overview": movie.get("overview", "Overview not available")}
                for movie in crew_movies
            ]

        # Top-rated movies in the same genre (using TMDb API)
        if movie_info["genres"]:
            primary_genre_id = movie_info["genres"][0]  # Use first genre
            top_rated = get_top_rated_by_genre(primary_genre_id)
            results["top_rated_genre"] = [
                {"title": movie["title"], "overview": movie["overview"], "rating": movie["vote_average"], "vote_count": movie["vote_count"]}
                for movie in top_rated
            ]

    else:
        st.subheader(f"❌ Movie '{movie_title}' not found. Showing top popular movies!")
        top_popular = df.sort_values("popularity", ascending=False).head(5)
        results["same_genre"] = [
            {"title": row["title"], "overview": row["overview"]}
            for _, row in top_popular.iterrows()
        ]

    return results

# --- Autocomplete Search Suggestions ---
def get_search_suggestions(query, movie_list, limit=5):
    """Generate movie title suggestions based on partial input."""
    if not query.strip():
        return []
    matches = process.extract(query.lower(), movie_list, scorer=fuzz.partial_ratio, limit=limit)
    return [match[0] for match in matches if match[1] > 70]  # Filter by similarity score

# --- Streamlit UI ---
def main():
    """Main function to run the Streamlit app."""
    st.title("🎬 AI Movie Recommender")
    st.markdown("Search for a movie to get recommendations by genre, crew, and top-rated movies in the same genre!")

    # Load data and compute similarity
    df = load_data()
    similarity = compute_similarity(df)
    movie_list = df["title"].str.lower().tolist()

    # Search bar with suggestions
    st.subheader("Search for a Movie")
    query = st.text_input("Enter a movie name:", "", key="search_input")
    suggestions = get_search_suggestions(query, movie_list)
    
    selected_movie = None
    if suggestions:
        suggestion_options = [""] + suggestions  # Include empty option
        selected_movie = st.selectbox("Suggestions:", suggestion_options, index=0, key="suggestions")
    
    # Use selected movie from suggestions if chosen, else use typed query
    movie_input = selected_movie if selected_movie else query

    if st.button("Recommend"):
        if movie_input.strip():
            results = recommend(movie_input, df, similarity)
            
            # Display searched movie
            if results["searched_movie"]:
                searched = results["searched_movie"]
                st.subheader(f"✅ Your Searched Movie: {searched['title']}")
                st.markdown(f"📖 **Overview:** {searched['overview']}")
                st.markdown(f"⭐ **Rating:** {searched['movie_info']['rating']} / 10 ({searched['movie_info']['vote_count']} votes)")
                st.markdown(f"🎭 **Genres:** {', '.join(searched['genres']) if searched['genres'] else 'N/A'}")
                st.markdown(f"🎬 **Director:** {searched['movie_info']['director']}")
                st.markdown(f"🌟 **Cast:** {', '.join(searched['movie_info']['cast']) if searched['movie_info']['cast'] else 'N/A'}")
                if searched["movie_info"]["poster_url"]:
                    st.image(searched["movie_info"]["poster_url"], caption=searched["title"], width=200)
                else:
                    st.image("https://via.placeholder.com/500x750.png?text=No+Poster+Available", caption="No Poster Available", width=200)

            # Display same genre recommendations
            if results["same_genre"]:
                st.subheader("🎥 Movies in the Same Genre")
                cols = st.columns(3)
                for i, movie in enumerate(results["same_genre"]):
                    with cols[i % 3]:
                        st.write(f"**👉 {movie['title']}**")
                        st.markdown(f"📖 **Overview:** {movie['overview']}")
                        movie_info = get_movie_info(movie['title'])
                        st.markdown(f"⭐ **Rating:** {movie_info['rating']} / 10 ({movie_info['vote_count']} votes)")
                        if movie_info["poster_url"]:
                            st.image(movie_info["poster_url"], caption=movie['title'], width=150)
                        else:
                            st.image("https://via.placeholder.com/500x750.png?text=No+Poster+Available", caption="No Poster Available", width=150)

            # Display same crew recommendations
            if results["same_crew"]:
                st.subheader("🎬 Movies by the Same Crew (Director)")
                cols = st.columns(3)
                for i, movie in enumerate(results["same_crew"]):
                    with cols[i % 3]:
                        st.write(f"**👉 {movie['title']}**")
                        st.markdown(f"📖 **Overview:** {movie['overview']}")
                        movie_info = get_movie_info(movie['title'])
                        st.markdown(f"⭐ **Rating:** {movie_info['rating']} / 10 ({movie_info['vote_count']} votes)")
                        if movie_info["poster_url"]:
                            st.image(movie_info["poster_url"], caption=movie['title'], width=150)
                        else:
                            st.image("https://via.placeholder.com/500x750.png?text=No+Poster+Available", caption="No Poster Available", width=150)

            # Display top-rated movies in the same genre
            if results["top_rated_genre"]:
                st.subheader("🏆 Top-Rated Movies in the Same Genre")
                cols = st.columns(3)
                for i, movie in enumerate(results["top_rated_genre"]):
                    with cols[i % 3]:
                        st.write(f"**👉 {movie['title']}**")
                        st.markdown(f"📖 **Overview:** {movie['overview']}")
                        st.markdown(f"⭐ **Rating:** {movie['rating']} / 10 ({movie['vote_count']} votes)")
                        poster_url = get_movie_info(movie["title"])["poster_url"]
                        if poster_url:
                            st.image(poster_url, caption=movie["title"], width=150)
                        else:
                            st.image("https://via.placeholder.com/500x750.png?text=No+Poster+Available", caption="No Poster Available", width=150)
        else:
            st.warning("Please enter a movie title!")

    # Top-rated movies (general)
    if st.button("Top Rated Movies"):
        top_movies = get_top_rated_by_genre("")  # Empty genre ID for general top-rated
        if top_movies:
            st.subheader("🎯 Top Rated Movies (All Genres)")
            cols = st.columns(3)
            for i, movie in enumerate(top_movies):
                with cols[i % 3]:
                    st.write(f"**👉 {movie['title']}**")
                    st.markdown(f"⭐ **Rating:** {movie['vote_average']} / 10 ({movie['vote_count']} votes)")
                    st.markdown(f"📖 **Overview:** {movie['overview']}")
                    poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else None
                    if poster_url:
                        st.image(poster_url, caption=movie["title"], width=150)
                    else:
                        st.image("https://via.placeholder.com/500x750.png?text=No+Poster+Available", caption="No Poster Available", width=150)

if __name__ == "__main__":
    main()
