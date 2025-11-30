import streamlit as st
import pandas as pd
import unicodedata
import re   # <-- REQUIRED for clean_name

from game_recommender import (
    clean_tags,
    rebuild_features,
    recommend_game,
    load_steam_data,
    load_user_data,
    extract_sentiment_and_percentage,
)

# ---------------------------------------
# NEW: CLEAN NAME FUNCTION
# ---------------------------------------
def clean_name(name: str) -> str:
    """Normalize Unicode, remove symbols like ®™, fix dashes, lowercase."""
    if not isinstance(name, str):
        return ""

    name = unicodedata.normalize("NFKD", name)  # normalize fancy chars

    # remove trademark symbols
    name = re.sub(r"[®©™]", "", name)

    # convert fancy dashes
    name = name.replace("–", "-").replace("—", "-")

    # collapse spaces
    name = re.sub(r"\s+", " ", name)

    return name.strip().lower()



# ---------------------------------------
# LOAD DATA (CACHED)
# ---------------------------------------
@st.cache_data
def load_steam():
    import os
    import pandas as pd

    # ---------- 1. Load main CSV EXACTLY like notebook ----------
    df = pd.read_csv("steam_games.csv", low_memory=False)

    # ---------- 2. Lowercase name ----------
    df['name_lower'] = df['name'].astype(str).str.lower().str.strip()

    # ---------- 3. Clean specific columns ----------
    cols_to_clean = [
        "desc_snippet",
        "minimum_requirements",
        "original_price",
        "popular_tags"
    ]
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"\.\.\.", "", regex=True)
                .str.strip()
            )

    # ---------- 4. Load user CSV and merge BEFORE processing ----------
    if os.path.exists("user_games.csv"):
        df_user = pd.read_csv("user_games.csv")
    else:
        df_user = pd.DataFrame(columns=df.columns)

    df = pd.concat([df, df_user], ignore_index=True)

    # ---------- 5. Lowercase name again ----------
    df["name_lower"] = df["name"].astype(str).str.lower().str.strip()

    # ---------- 5.5 ADD CLEAN NAME COLUMN ----------
    df["name_clean"] = df["name"].astype(str).apply(clean_name)

    # ---------- 6. Keep ONLY these exact columns ----------
    columns_to_keep = [
        "name",
        "all_reviews",
        "popular_tags",
        "name_lower",
        "genre",
        "developer",
        "publisher",
        "desc_snippet",
        "minimum_requirements",
        "original_price",
        "name_clean",            # <-- ADD THIS
    ]
    df = df[columns_to_keep].copy()

    # ---------- 7. Parse popular_tags EXACTLY like notebook ----------
    df["popular_tags"] = (
        df["popular_tags"]
        .astype(str)
        .str.replace(r"[\[\]'\" ]", "", regex=True)
        .str.split(",")
    )

    # ---------- 8. Same genre fallback logic ----------
    df["genre"] = df["genre"].fillna(
        df["popular_tags"].apply(
            lambda tags: tags[0] if isinstance(tags, list) and len(tags) > 0 else None
        )
    )

    # ---------- 9. Remove invalid rows ----------
    df = df[df["name"].notna()]
    df = df[df["popular_tags"].astype(str) != "[]"]

    # ---------- 10. EXACTLY like notebook: drop all_reviews ----------
    df = extract_sentiment_and_percentage(df, "all_reviews")

    df = df.drop(columns=["all_reviews"])

    # ---------- 12. Build clean_tags ----------
    df["clean_tags"] = df["popular_tags"].apply(lambda lst: clean_tags(" ".join(lst)))

    # ---------- 13. Rebuild features EXACTLY the same ----------
    df, vectorizer, feature_matrix = rebuild_features(df)

    return df, df_user, vectorizer, feature_matrix


df, df_user, vectorizer, feature_matrix = load_steam()



# ---------------------------------------
# UI
# ---------------------------------------
st.title("Game Recommendation Engine")

user_title = st.text_input("Enter a game name:")

if user_title:

    # NEW: CLEAN USER INPUT NAME
    title_clean = clean_name(user_title)

    # Build sets of cleaned names
    steam_names = set(df["name_clean"])
    user_names = set(df_user["name_lower"].apply(clean_name)) if len(df_user) > 0 else set()

    all_names = steam_names | user_names

    # -----------------------------
    # CASE A: game is known (Steam or user)
    # -----------------------------
    if title_clean in all_names:
        st.subheader(f"Recommendations for '{user_title}'")

        try:
            results = recommend_game(
                user_title,
                df,
                df_user,
                vectorizer,
                feature_matrix,
            )
        except Exception as e:
            st.error(f"❌ Error in recommend_game(): {e}")
            results = []

        if results:
            for game in results:
                st.markdown(f"### 🎮 {game['name']}")
                st.write(f"**Genre:** {game.get('genre', 'N/A')}")
                st.write(f"**Developer:** {game.get('developer', 'N/A')}")
                if game.get("percentage") is not None:
                    st.write(
                        f"**Reviews:** {game.get('sentiment', 'N/A')} "
                        f"({game.get('percentage')}%)"
                    )
                else:
                    st.write(f"**Reviews:** {game.get('sentiment', 'N/A')}")
                st.write(f"**Price:** {game.get('price', 'N/A')}")
                st.write("---")
        else:
            st.warning("No recommendations found after filtering.")

    # -----------------------------
    # CASE B: completely unknown → let user add game
    # -----------------------------
    else:
        st.warning("Game not found in database. Add it below:")

        with st.form("add_game_form"):
            genre = st.text_input("Genre")
            dev = st.text_input("Developer")
            tags = st.text_input("Tags (comma-separated)")
            submitted = st.form_submit_button("Add Game")

        if submitted:
            raw_tags = [t.strip() for t in tags.split(",") if t.strip()]

            cleaned_tags_string = clean_tags(" ".join(raw_tags).lower())

            new_row = {
                "name": user_title,
                "name_lower": user_title.lower().strip(),
                "genre": genre,
                "developer": dev,
                "popular_tags": raw_tags,
                "clean_tags": cleaned_tags_string,
                "sentiment": "No reviews",
                "percentage": None,
                "original_price": "",
                "name_clean": title_clean,   # <-- ADD THIS
            }

            df_user.loc[len(df_user)] = new_row
            df_user.to_csv("user_games.csv", index=False)

            st.success(f"'{user_title}' added successfully! Now you can search it again.")
            st.cache_data.clear()
            st.rerun()