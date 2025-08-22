# Transformer Library Examples

This project demonstrates how to use two key classes from the [SentenceTransformers](https://www.sbert.net/) library:

- **SentenceTransformer**: Generate embeddings (vector representations) for sentences.
- **CrossEncoder**: Directly compute similarity/relevance scores between pairs of sentences.

---

## 🚀 Quick Overview

- `SentenceTransformer` → **Fast, scalable** → Encode sentences to embeddings → Good for **search, clustering, recommendations**.
- `CrossEncoder` → **Accurate, but slower** → Compare two texts directly → Good for **ranking or pairwise matching**.

---

## 1. SentenceTransformer

The `SentenceTransformer` converts text into **dense embeddings** (lists of numbers). These embeddings can be used for:

- Semantic search
- Clustering
- Recommendations
- Measuring sentence similarity (using cosine similarity)

### Example
```python
from sentence_transformers import SentenceTransformer, util

# Load a pre-trained model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode sentences
sentences = ["I love machine learning", "I enjoy artificial intelligence"]
embeddings = model.encode(sentences)

# Compute similarity
similarity = util.cos_sim(embeddings[0], embeddings[1])
print(f"Similarity: {similarity.item():.4f}")
```

---

## 2. CrossEncoder

The `CrossEncoder` takes **two texts as input at the same time** and directly predicts a **score** for their relationship (e.g., similarity, relevance).

It is slower than SentenceTransformer (because it processes both sentences together), but usually more accurate for pairwise comparisons.

### Example
```python
from sentence_transformers import CrossEncoder

# Load model
model = CrossEncoder("cross-encoder/stsb-roberta-base")

# Predict similarity score
pairs = [("I love AI", "I enjoy artificial intelligence")]
scores = model.predict(pairs)

print(f"Similarity Score: {scores[0]:.4f}")
```

---

## 3. Using Both Together (Retrieval + Reranking)

A common real-world setup is:

1. Use **SentenceTransformer** to encode your documents and quickly retrieve top-k candidates.  
2. Use **CrossEncoder** to rerank those candidates for higher accuracy.  

### Example
```python
from sentence_transformers import SentenceTransformer, CrossEncoder, util

# Step 1: Embed documents with SentenceTransformer
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
documents = [
    "Python is a programming language",
    "Transformers are neural network models",
    "Artificial Intelligence is the future",
    "I love playing football"
]

doc_embeddings = bi_encoder.encode(documents, convert_to_tensor=True)

# Step 2: Query
query = "What is AI?"
query_embedding = bi_encoder.encode(query, convert_to_tensor=True)

# Step 3: Retrieve top-k documents
hits = util.semantic_search(query_embedding, doc_embeddings, top_k=3)[0]

# Step 4: Rerank with CrossEncoder
cross_encoder = CrossEncoder("cross-encoder/stsb-roberta-base")
pairs = [(query, documents[hit['corpus_id']]) for hit in hits]
rerank_scores = cross_encoder.predict(pairs)

# Combine results
for (doc, score) in zip(pairs, rerank_scores):
    print(f"Query: {doc[0]} | Document: {doc[1]} | Score: {score:.4f}")
```

---

## ⚡ Key Differences

| Feature / Use Case       | SentenceTransformer                 | CrossEncoder                |
|---------------------------|--------------------------------------|-----------------------------|
| **Input**                 | Single sentence                     | Sentence pair               |
| **Output**                | Embedding vector                    | Single similarity/relevance score |
| **Speed**                 | Fast (good for millions of docs)    | Slower (good for small comparisons) |
| **Best for**              | Search, clustering, recommendations | Reranking, QA matching, pairwise accuracy |

---

## 🛠 Installation

```bash
pip install sentence-transformers
```

---

## 📚 References

- [SentenceTransformers Documentation](https://www.sbert.net/)  
- [Pretrained Models on HuggingFace](https://huggingface.co/models?library=sentence-transformers)  
