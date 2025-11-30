import pandas as pd
import numpy as np
import re
import unicodedata
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_sentiment_and_percentage(df, column_name):
    """Compact version using str.extract()"""
    
    # Extract sentiment (everything before first comma)
    df['sentiment'] = df[column_name].str.extract(r'^([^,]+)', expand=False).str.strip()
    
    # Extract percentage (digits before %)
    df['percentage'] = df[column_name].str.extract(r'(\d+)%', expand=False).astype('Int64')
    
    return df

def clean_name(name: str) -> str:
    """Normalize title so user input matches dataset names (removes ®, fancy dashes, etc.)."""
    if not isinstance(name, str):
        return ""

    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r"[®©™]", "", name)           # drop trademark symbols
    name = name.replace("–", "-").replace("—", "-")  # fancy dashes → normal -
    name = re.sub(r"\s+", " ", name)            # collapse spaces
    return name.strip().lower()

def clean_tags(tag_list):
    """
    Convert list-like string ['FPS','CoOp','MilitaryShooter']
    into clean, spaced tokens: 'fps coop military shooter'
    """
    if not isinstance(tag_list, str):
        return ""

    # Remove brackets and quotes
    tag_list = tag_list.replace("[", "").replace("]", "").replace("'", "").replace('"', "")

    # Split by comma
    tags = [t.strip() for t in tag_list.split(",")]

    clean_tokens = []
    for tag in tags:
        # Split CamelCase (e.g., MilitaryShooter → Military Shooter)
        separated = re.sub(r'(?<!^)(?=[A-Z])', ' ', tag)
        clean_tokens.append(separated.lower())

    return " ".join(clean_tokens)

def rebuild_features(df):
    # Convert tags list → clean string
    #df["clean_tags"] = df["popular_tags"].apply(
    #    lambda x: " ".join(x) if isinstance(x, list) else str(x)
    #)

    df["clean_tags"] = df["popular_tags"].apply(
        lambda x: clean_tags(" ".join(x).lower()) if isinstance(x, list) else clean_tags(str(x).lower())
    )

    #df["clean_tags"] = df["popular_tags"].apply(clean_tags)

    # Weighted feature combination 
    df["Feature_Text"] = (
        df["genre"].fillna("").astype(str) + " " +
        #df["developer"].fillna("").astype(str) + " " +
        (df["clean_tags"].fillna("").astype(str) + " ") * 3
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 1),
        min_df=2
    )

    feature_matrix = vectorizer.fit_transform(df["Feature_Text"])

    return df, vectorizer, feature_matrix

def same_franchise(title1, title2):
    import re

    t1 = title1.lower().strip()
    t2 = title2.lower().strip()

    # Remove ALL non-alphanumeric characters
    t1 = re.sub(r'[^a-z0-9\s]', ' ', t1)
    t2 = re.sub(r'[^a-z0-9\s]', ' ', t2)

    # Tokenize meaningful words
    words1 = [w for w in t1.split() if len(w) > 2]
    words2 = [w for w in t2.split() if len(w) > 2]

    if len(words1) < 2 or len(words2) < 2:
        return False

    prefix1 = " ".join(words1[:2])
    prefix2 = " ".join(words2[:2])

    return prefix1 == prefix2

def recommend_game(title, df, df_user, vectorizer, feature_matrix, top_n=5):
    df = df.reset_index(drop=True)

    # ensure we have a clean-name column (works with or without frontend’s preprocessing)
    if "name_clean" not in df.columns:
        df["name_clean"] = df["name"].astype(str).apply(clean_name)

    # use cleaned title for matching (handles ®, ™, fancy dashes, etc.)
    title_clean = clean_name(title)
    match = df[df["name_clean"] == title_clean]

    # still keep lowercased name for your user-game logic later
    title_lower = title.lower().strip()
    df["name_lower"] = df["name"].astype(str).str.lower().str.strip()

    # if somehow nothing matches, just behave like before (no recs in Streamlit)
    if match.empty:
        return []

    # use the index from the clean-name match
    idx = match.index[0]

    sims = cosine_similarity(feature_matrix[idx], feature_matrix)[0]
    quality_weight = df["percentage"].fillna(70) / 100
    sims = sims * quality_weight

    # Set ultra-low similarities to zero to remove noise
    sims[sims < 0.02] = 0

    ranked = sims.argsort()[::-1]

    results = []
    seen = set()

    # Prepare tag sets for quality filtering
    target_tags = set(df.iloc[idx]["clean_tags"].split())

    user_games_set = set(df_user["name_lower"].dropna().tolist())
    target_is_user_added = title_lower in user_games_set

    # Identify if target is a AAA shooter (CoD-type)
    target_is_shooter = (
        "shooter" in target_tags or 
        "fps" in target_tags or
        "first-person" in target_tags
    )

    for i in ranked:
        if i == idx:
            continue

        # skip if similarity was wiped out
        if sims[i] == 0:
            continue

        candidate = df.iloc[i]
        candidate_tags = set(candidate["clean_tags"].split())

        is_user_added = candidate["name_lower"] in user_games_set

        if is_user_added:
            continue

        # ---------------------------------------------------------------
        # A. Skip if NO shared tags at all (prevents irrelevant garbage)
        # ---------------------------------------------------------------
        if len(target_tags & candidate_tags) == 0:
            continue

        # ---------------------------------------------------------------
        # REVIEW FILTER — ONLY apply this when TARGET is a Steam game
        # ---------------------------------------------------------------
        if not target_is_user_added:
            # If candidate is from Steam dataset, apply review filter
            if not is_user_added:
                if (
                    candidate["sentiment"] in (None, "", "No reviews")
                    or pd.isna(candidate["percentage"])
                ):
                    continue

        # ---------------------------------------------------------------
        # B. Skip same franchise titles (your original rule)
        # ---------------------------------------------------------------
        if same_franchise(title, candidate["name"]):
            continue

        # ---------------------------------------------------------------
        # C. AAA shooter → avoid low-quality indie shooters
        # ---------------------------------------------------------------
        if target_is_shooter:
            # If it’s free-to-play, mixed reviews, OR < 60% → skip
            if (
                ("indie" in candidate_tags and "free" in str(candidate["original_price"]).lower())
                or (candidate["sentiment"] == "Mixed")
                or (pd.notna(candidate["percentage"]) and candidate["percentage"] < 60)
            ):
                continue

        # ---------------------------------------------------------------
        # D. Avoid duplicates (your rule)
        # ---------------------------------------------------------------
        if candidate["name"] in seen:
            continue

        # Passed all filters → add to results
        results.append(i)
        seen.add(candidate["name"])

        if len(results) == top_n:
            break

    # ---------------------------------------------------------------
    # INSTEAD OF PRINTING → BUILD LIST[DICT] FOR STREAMLIT
    # ---------------------------------------------------------------
    output = []

    for i in results:
        row = df.iloc[i]

        genre_val = row["genre"]
        if isinstance(genre_val, list):
            genre = ", ".join(genre_val)
        else:
            genre = str(genre_val) if pd.notna(genre_val) else "Unknown"

        dev = row["developer"] if isinstance(row["developer"], str) else ""
        sentiment = row["sentiment"] if isinstance(row["sentiment"], str) else "No reviews"
        percent = int(row["percentage"]) if pd.notna(row["percentage"]) else None

        # You used original_price in your prints, so we keep that first
        price_val = row.get("original_price", None)
        if isinstance(price_val, str) and price_val:
            price = price_val
        else:
            # Fallback if you later add a 'price' column
            price = row["price"] if "price" in row.index else "Unknown"

        output.append({
            "name": row["name"],
            "genre": genre,
            "developer": dev,
            "sentiment": sentiment,
            "percentage": percent,
            "price": price,
        })

    return output

def save_user_game(df_user, path="user_games.csv"):
    df_user.to_csv(path, index=False)

def load_user_data(path="user_games.csv"):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=[
            "name", "name_lower", "genre", "developer",
            "popular_tags", "clean_tags", "sentiment",
            "percentage", "original_price"
        ])
    df["name_lower"] = df["name"].str.lower()
    return df

def load_steam_data(path="steam_games.csv"):
    df = pd.read_csv(path)
    
    # Normalize name
    df["name_lower"] = df["name"].astype(str).str.lower().str.strip()
    
    # Clean tags BEFORE rebuild_features
    df["clean_tags"] = df["popular_tags"].apply(clean_tags)
    
    # Build features
    df, vect, matrix = rebuild_features(df)
    
    return df, vect, matrix