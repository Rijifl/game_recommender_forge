import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#loading dataset
df = pd.read_csv("vgsales.csv")

#loads user added games file if it exists
if os.path.exists("user_games.csv"):
    df_memory = pd.read_csv("user_games.csv")
else:
    df_memory = pd.DataFrame(columns=["Name", "Platform", "Genre", "Publisher"])

#combines datasets
df = pd.concat([df, df_memory], ignore_index=True)

#descriptive columns
df = df[['Name', 'Platform', 'Year', 'Genre', 'Publisher']]

#missing values handling
df = df.fillna('Unknown')

#makes each game description into a string
df['features'] = (
    df['Platform'].astype(str) + ' ' +
    df['Genre'].astype(str) + ' ' +
    df['Publisher'].astype(str)
)

#text to numbers
vectorizer = TfidfVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(df['features'])

similarity = cosine_similarity(feature_matrix)

#functions !!

#recommendation function
def recommend_game(title):
    global df_memory

    lower_title = title.lower()
    df['name_lower'] = df['Name'].str.lower()

    #adding new games
    if lower_title not in df['name_lower'].values:
        print("Game not found in dataset.")
        print("Let's add it! Please provide the following details:\n")

        platform = input("Platform: ")
        genre = input("Genre: ")
        publisher = input("Publisher: ")

        new_row = {
        "Name": title,
        "Platform": platform,
        "Genre": genre,
        "Publisher": publisher
        }

        df_memory = pd.concat([df_memory, pd.DataFrame([new_row])], ignore_index=True)
        df_memory.to_csv("user_games.csv", index=False)

        df.loc[len(df)] = new_row

        print(f"\n'{title}' has been added to the games list!")

        new_game = f"{platform} {genre} {publisher}"

        game_vector = vectorizer.transform([new_game])

        game_similarity = cosine_similarity(game_vector, feature_matrix)[0]
        score_indexed = game_similarity.argsort()[-5:][::-1]

        print(f"\n Games similar to '{title}':")
        for i in score_indexed:
            print(f"- {df.iloc[i]['Name']} ({df.iloc[i]['Genre']}, {df.iloc[i]['Platform']})")
                
        return
    

    #finding similar games for existing games in dataset
    idx = df[df['Name'] == title].index[0]
    scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    print(f" Games similar to '{title}':")
    for i, score in sorted_scores:
        print(f"- {df.iloc[i]['Name']} ({df.iloc[i]['Genre']}, {df.iloc[i]['Platform']})")
    

user_input = input("🎮 Enter a game title: ")

recommend_game(user_input)