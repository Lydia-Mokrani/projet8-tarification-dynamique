import pandas as pd

def load_real_reviews(path, text_col, rating_col, n_max=10000, seed=42):
    df = pd.read_csv(path)[[text_col, rating_col]].dropna()
    df.columns = ["text", "rating"]
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna()
    df = df[df.rating != 3].drop_duplicates(subset="text")
    df["label"] = (df.rating >= 4).astype(int)
    n = min(n_max // 2, df.label.value_counts().min())
    df = pd.concat([g.sample(n, random_state=seed) for _, g in df.groupby("label")])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
