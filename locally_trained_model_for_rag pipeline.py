import numpy as np
import faiss
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer

# --------------------------
# 1. Sample Dataset (IMDB reviews-like)
# --------------------------
docs = [
    "I loved the movie, it was fantastic and well-acted!",
    "Terrible movie, I hated every minute of it.",
    "An excellent performance by the lead actor.",
    "The film was boring and too long.",
    "Absolutely wonderful and inspiring story.",
    "Worst movie ever made, don’t waste your time."
]
labels = [1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative

# --------------------------
# 2. Train Classifier (FFNN)
# --------------------------

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(docs).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Classifier accuracy:", accuracy_score(y_test, y_pred))

# --------------------------
# 3. Build Embeddings for RAG
# --------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, free model
doc_embeddings = embedder.encode(docs)

# --------------------------
# 4. Create FAISS Vector DB
# --------------------------
dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)  # simple in-memory index
index.add(np.array(doc_embeddings))  # store all docs

# --------------------------
# 5. Retrieval + Classification
# --------------------------
def rag_query(query, top_k=2):
    # Step A: Embed the query
    q_emb = embedder.encode([query])

    # Step B: Search in FAISS
    _, indices = index.search(np.array(q_emb), top_k)
    retrieved_docs = [docs[i] for i in indices[0]]

    # Step C: Classify retrieved docs
    preds = []
    for d in retrieved_docs:
        vec = vectorizer.transform([d]).toarray()
        label = clf.predict(vec)[0]
        preds.append((d, "Positive" if label == 1 else "Negative"))

    return preds

# --------------------------
# 6. Test the RAG system
# --------------------------
query = "Was the movie good or bad?"
results = rag_query(query)

print("\n🔍 Query:", query)
print("Retrieved docs + Predicted labels:")
for doc, label in results:
    print(f"- {doc} => {label}")
