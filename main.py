import pandas as pd

from sklearn.feature\_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine\_similarity



# Load dataset



df = pd.read\_csv("merged\_movies.csv")

df.columns = df.columns.str.strip()



# TF-IDF for recommendations



tfidf = TfidfVectorizer(stop\_words="english")

vector = tfidf.fit\_transform(df\["overview"].fillna(""))

similarity = cosine\_similarity(vector)



# Recommendation function



def recommend(movie\_title):

movie\_title\_lower = movie\_title.lower()

movie\_list = df\["title"].str.lower().tolist()



```

if movie_title_lower in movie_list:

    idx = movie_list.index(movie_title_lower)



    # Show the searched movie's details

    searched_movie_title = df.loc[idx, "title"]

    searched_movie_overview = df.loc[idx, "overview"]

    print(f"✅ Your searched movie: {searched_movie_title}")

    print(f"📖 Overview: {searched_movie_overview}\n")

    

    recommended_movies = sorted(

        list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True

    )[1:6]



    for i in recommended_movies:

        print(f"👉 {df.loc[i[0], 'title']}")

        print(f"📖 {df.loc[i[0], 'overview']}\n")

else:

    print(f"❌ Movie '{movie_title}' not found. Showing top popular movies!\n")

    top_popular = df.sort_values("popularity", ascending=False).head(5)



    for _, row in top_popular.iterrows():

        print(f"👉 {row['title']}")

        print(f"📖 {row['overview']}\n")

```



# Example usage



if **name** == "**main**":

recommend("Baahubali: The Beginning")

recommend("Unknown Movie")
