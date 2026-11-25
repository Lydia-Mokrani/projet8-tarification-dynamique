from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from ecom_agents.agents.reputation import NaiveBayesSentiment, TransformerSentiment
from real_reviews import load_real_reviews

df = load_real_reviews("Reviews.csv", "Text", "Score")
Xtr, Xte, ytr, yte = train_test_split(
    df.text.to_numpy(), df.label.to_numpy(),
    test_size=0.2, random_state=42, stratify=df.label)

nb = NaiveBayesSentiment().fit(Xtr, ytr)
bert = TransformerSentiment("torch").fit(Xtr, ytr)

for name, m in (("Naive Bayes", nb), ("DistilBERT", bert)):
    pred = (m.predict_proba(Xte) > 0.5).astype(int)
    print(name, "accuracy", round(accuracy_score(yte, pred), 3),
          "f1", round(f1_score(yte, pred), 3))