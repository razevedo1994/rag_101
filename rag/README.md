# RAG (Retrieval-Augmented Generation) Project Documentation

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Mathematical Concepts](#mathematical-concepts)
  - [1. Vector Embeddings](#1-vector-embeddings)
  - [2. Cosine Similarity](#2-cosine-similarity)
  - [3. Vector Norms (L2 Norm)](#3-vector-norms-l2-norm)
  - [4. Dot Product](#4-dot-product)
- [Pipeline Flow](#pipeline-flow)
- [Implementation Details](#implementation-details)
- [Usage](#usage)
- [Dependencies](#dependencies)

---

## Overview

This project implements a complete **Retrieval-Augmented Generation (RAG)** pipeline in Python, demonstrating how semantic search and context-aware text generation work together. The system:

1. **Embeds documents** into a continuous vector space using transformer models
2. **Retrieves relevant documents** using cosine similarity for semantic search
3. **Generates natural language answers** using an LLM (LLaMA 3.1) grounded in retrieved context

**Corpus**: 11 Portuguese sentences about machine learning concepts, applications, and techniques.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    1. OFFLINE INDEXING                       │
│  Documents → Embedding Model → Document Embeddings (11×384)  │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                    2. QUERY EMBEDDING                        │
│      User Query → Embedding Model → Query Vector (384)       │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                    3. SIMILARITY SEARCH                      │
│   Query Vector × Doc Embeddings → Cosine Similarities        │
│   Sort by similarity → Retrieve Top-K (default: 3)           │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                    4. CONTEXT ASSEMBLY                       │
│     Retrieved Documents → Concatenate into Context String    │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                    5. ANSWER GENERATION                      │
│   LLM (LLaMA 3.1) + Context + Query → Natural Language Answer│
└─────────────────────────────────────────────────────────────┘
```

---

## Mathematical Concepts

### 1. Vector Embeddings

**Definition**: Dense vector representations of text in a continuous n-dimensional vector space, where semantic meaning is encoded in geometric relationships.

**Model**: `all-MiniLM-L6-v2` (SentenceTransformer)
- **Architecture**: 6-layer BERT-based transformer
- **Embedding Dimension**: 384
- **Training**: Contrastive learning for semantic similarity

**Mathematical Representation**:

For a text document d, the embedding function E maps text to a vector:

```
E: Text → ℝ³⁸⁴
E(d) = [e₁, e₂, e₃, ..., e₃₈₄]
```

Where each eᵢ ∈ ℝ is a real-valued component capturing semantic features.

**Properties**:
- Similar meanings → vectors close in embedding space
- Semantic relationships preserved as geometric relationships
- Distance/angle between vectors correlates with semantic similarity

**Implementation**:
```python
doc_embeddings = model.encode(documents)  # Shape: (11, 384)
query_embedding = model.encode([query])[0]  # Shape: (384,)
```

---

### 2. Cosine Similarity

**Definition**: Measures the cosine of the angle between two non-zero vectors, quantifying their directional similarity independent of magnitude.

**Formula**:

```
                    a · b
cos(θ) = ────────────────────────
         ‖a‖ × ‖b‖

where:
  a · b    = dot product (a₁b₁ + a₂b₂ + ... + aₙbₙ)
  ‖a‖      = L2 norm of vector a = √(a₁² + a₂² + ... + aₙ²)
  ‖b‖      = L2 norm of vector b = √(b₁² + b₂² + ... + bₙ²)
  θ        = angle between vectors a and b
```

**Alternative Formulation**:

```
           n
          Σ aᵢbᵢ
          i=1
cos(θ) = ─────────────────────
         √(Σaᵢ²) × √(Σbᵢ²)
```

**Range**: [-1, 1]
- **1**: Vectors point in identical direction (most similar)
- **0**: Vectors are orthogonal/perpendicular (unrelated)
- **-1**: Vectors point in opposite directions (most dissimilar)

**Why Cosine for Text?**
- **Scale-invariant**: Measures angle, not magnitude (important since embeddings may have varying lengths)
- **Normalized similarity**: Natural interpretation as correlation
- **Computationally efficient**: Simple vector operations

**Implementation**:
```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

**Geometric Interpretation**:

```
        b
        ↑
       /|
      / |
     /  |θ
    /   |
   /    |
  ────────→ a
  
  cos(θ) measures angle θ between a and b
  Smaller θ → larger cos(θ) → more similar
```

---

### 3. Vector Norms (L2 Norm)

**Definition**: The L2 (Euclidean) norm measures the "length" or magnitude of a vector in n-dimensional space.

**Formula**:

```
‖a‖₂ = √(a₁² + a₂² + ... + aₙ²)

     = √(Σ aᵢ²)  for i=1 to n
       i
```

**For 384-dimensional embeddings**:

```
‖a‖₂ = √(a₁² + a₂² + a₃² + ... + a₃₈₄²)
```

**Properties**:
- Always non-negative: ‖a‖ ≥ 0
- Only zero vector has norm 0: ‖a‖ = 0 ⟺ a = 0
- Triangle inequality: ‖a + b‖ ≤ ‖a‖ + ‖b‖
- Scalar multiplication: ‖ka‖ = |k|·‖a‖

**Purpose in RAG**:
- Normalizes vectors for cosine similarity calculation
- Enables scale-invariant comparison
- Converts dot product to angular similarity

**Implementation**:
```python
np.linalg.norm(a)  # Computes L2 norm
```

**Related Norm (L1 - Manhattan Distance)**:

```
‖a‖₁ = |a₁| + |a₂| + ... + |aₙ| = Σ|aᵢ|
```

---

### 4. Dot Product

**Definition**: Algebraic operation that takes two equal-length sequences of numbers and returns a single scalar value.

**Formula**:

```
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ

      n
    = Σ aᵢbᵢ
     i=1
```

**For 384-dimensional vectors**:

```
a · b = Σ aᵢbᵢ  for i=1 to 384
        i
```

**Geometric Interpretation**:

```
a · b = ‖a‖ × ‖b‖ × cos(θ)

where θ is the angle between vectors a and b
```

This reveals why dot product appears in cosine similarity:

```
            a · b              ‖a‖ × ‖b‖ × cos(θ)
cos(θ) = ───────────  =  ────────────────────  = cos(θ)
         ‖a‖ × ‖b‖            ‖a‖ × ‖b‖
```

**Properties**:
- **Commutative**: a · b = b · a
- **Distributive**: a · (b + c) = a · b + a · c
- **Scalar multiplication**: (ka) · b = k(a · b)
- **Positive when θ < 90°**: Vectors point "same general direction"
- **Negative when θ > 90°**: Vectors point "opposite general direction"
- **Zero when θ = 90°**: Vectors are orthogonal

**Purpose in RAG**:
- Numerator of cosine similarity
- Measures alignment between query and document vectors
- Higher values indicate stronger semantic relationship

**Implementation**:
```python
np.dot(a, b)  # Efficient vectorized computation
```

---

## Pipeline Flow

### Phase 1: Initialization (Offline)

```python
# Load documents
documents = [
    "Machine learning é uma área da inteligência artificial...",
    # ... 10 more documents
]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode all documents into vectors
doc_embeddings = model.encode(documents)  # Shape: (11, 384)
```

**Mathematical Operation**:
```
D = {d₁, d₂, ..., d₁₁}  (document set)
E_D = {E(d₁), E(d₂), ..., E(d₁₁)}  (embedding set)
where E(dᵢ) ∈ ℝ³⁸⁴
```

---

### Phase 2: Query Processing

```python
query = "O que é machine learning?"
query_embedding = model.encode([query])[0]  # Shape: (384,)
```

**Mathematical Operation**:
```
q ∈ Text
E(q) ∈ ℝ³⁸⁴
```

---

### Phase 3: Similarity Search

```python
similarities = []
for i, doc_emb in enumerate(doc_embeddings):
    sim = cosine_similarity(query_embedding, doc_emb)
    similarities.append((i, sim))

similarities.sort(key=lambda x: x[1], reverse=True)
top_docs = [(documents[i], sim) for i, sim in similarities[:3]]
```

**Mathematical Operation**:

For each document embedding vᵢ:

```
sᵢ = cosine_similarity(E(q), E(dᵢ))

     E(q) · E(dᵢ)
  = ──────────────────
    ‖E(q)‖ × ‖E(dᵢ)‖

Retrieve: top-k documents by sᵢ (descending)
```

**Complexity**: O(n × d)
- n = number of documents (11)
- d = embedding dimension (384)
- 11 cosine similarity calculations, each O(384)

---

### Phase 4: Context Assembly

```python
context = "\n".join([doc for doc, _ in top_docs])
```

Concatenates top-k retrieved documents into a single context string.

---

### Phase 5: Answer Generation

```python
prompt = f"Contexto:\n{context}\n\nPergunta: {query}"

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "Você é um especialista em machine learning..."},
        {"role": "user", "content": prompt}
    ],
    temperature=0
)

answer = response.choices[0].message.content
```

**LLM Operation**:
- Input: System prompt + context + query
- Model: LLaMA 3.1 (8B parameters, decoder-only transformer)
- Temperature: 0 (deterministic, factual responses)
- Output: Natural language answer grounded in retrieved context

---

## Implementation Details

### File: `rag.py`

**Core Functions**:

1. **`cosine_similarity(a, b)`**
   - Computes cosine similarity between two vectors
   - Returns scalar in range [-1, 1]

2. **`retrieve(query, top_k=3)`**
   - Embeds query
   - Computes similarity with all documents
   - Returns top-k most similar documents with scores

3. **`generate_answer(query, retrieve_docs)`**
   - Formats retrieved documents as context
   - Constructs prompt with system instructions
   - Calls LLM API (Groq/LLaMA 3.1)
   - Returns generated answer

4. **`rag(query, top_k=3)`**
   - Main pipeline function
   - Combines retrieval and generation
   - Returns structured response with answer and source documents

**Models Used**:

| Component | Model | Purpose | Details |
|-----------|-------|---------|---------|
| Embeddings | all-MiniLM-L6-v2 | Text → Vector | 384-dim, BERT-based, 6 layers, 22M params |
| LLM | llama-3.1-8b-instant | Vector → Text | 8B params, decoder-only, via Groq API |

**Distance Metric**: Cosine similarity (angular distance in embedding space)

---

## Usage

### Basic Example

```python
# Run a single query
result = rag("O que é machine learning?")
print(result['answer'])
print(result['retrieve_docs'])
```

### Retrieve Only

```python
# Get relevant documents without generation
docs = retrieve("O que é machine learning?", top_k=3)
for doc, score in docs:
    print(f"[Score: {score:.4f}] {doc}")
```

### Custom Top-K

```python
# Retrieve more/fewer documents
result = rag("Como funciona aprendizado supervisionado?", top_k=5)
```

---

## Dependencies

```txt
numpy>=2.4.2           # Numerical operations (dot product, norms)
sentence-transformers>=5.2.2  # Embedding model (SentenceTransformer)
groq>=1.0.0           # LLM API client (Groq/LLaMA)
torch                 # PyTorch backend for transformers
transformers          # HuggingFace transformers library
```

### Installation

```bash
pip install numpy sentence-transformers groq
```

### Environment Setup

Create a `.env` file with your Groq API key:

```
GROQ_API_KEY=your_api_key_here
```

---

## Advanced Concepts (Implicit in Implementation)

### Transformer Attention Mechanism

The embedding model uses self-attention to compute contextual representations:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

where:
  Q = Query matrix
  K = Key matrix
  V = Value matrix
  d_k = dimension of key vectors (scaling factor)
```

This allows the model to capture semantic relationships between words in context.

---

### Contrastive Learning (Model Training)

The SentenceTransformer model was trained using contrastive loss to optimize embeddings:

```
L_contrastive = - log( e^(sim(a,p)/τ) / Σ_n e^(sim(a,n)/τ) )

where:
  a = anchor (query)
  p = positive (similar sentence)
  n = negatives (dissimilar sentences)
  τ = temperature parameter
  sim = cosine similarity
```

This encourages similar sentences to have similar embeddings while pushing dissimilar sentences apart.

---

### Vector Space Properties

**Embeddings form a semantic space where**:

1. **Distance correlates with similarity**:
   ```
   semantic_similarity(text_a, text_b) ≈ cosine_similarity(E(text_a), E(text_b))
   ```

2. **Semantic operations are geometric**:
   ```
   E("king") - E("man") + E("woman") ≈ E("queen")
   ```

3. **Clusters form around concepts**:
   - Documents about "supervised learning" cluster together
   - Documents about "neural networks" form another cluster

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Details |
|-----------|------------|---------|
| Encode document | O(L × d²) | L = sequence length, d = hidden dim (384) |
| Cosine similarity | O(d) | Single vector comparison (384 ops) |
| Retrieve (linear scan) | O(n × d) | n = corpus size (11), d = embedding dim |
| LLM generation | O(context_length × layers) | Depends on Groq API |

### Scalability Considerations

**Current Implementation**:
- **Linear scan**: O(n) retrieval
- **Works well** for small corpora (< 10K documents)

**For Large Scale** (> 100K documents):
- Use **vector databases** (Pinecone, Weaviate, Qdrant, Milvus)
- Implement **Approximate Nearest Neighbor (ANN)** search:
  - **FAISS** (Facebook AI Similarity Search)
  - **HNSW** (Hierarchical Navigable Small World graphs)
  - **IVF** (Inverted File Index)
- Complexity reduction: O(n) → O(log n) or O(√n)

---

## Potential Extensions

### 1. Alternative Distance Metrics

**Euclidean Distance**:
```
d_euclidean(a, b) = ‖a - b‖ = √(Σ(aᵢ - bᵢ)²)
```

**Manhattan Distance**:
```
d_manhattan(a, b) = Σ|aᵢ - bᵢ|
```

**Dot Product (unnormalized)**:
```
similarity_dot(a, b) = a · b = Σ(aᵢbᵢ)
```

### 2. Hybrid Search

Combine dense (embedding) and sparse (BM25, TF-IDF) retrieval:

```
score_hybrid = α × score_dense + (1-α) × score_sparse
```

### 3. Re-ranking

Add a cross-encoder model to re-rank retrieved documents:

```
query + doc → cross_encoder → relevance_score
```

### 4. Evaluation Metrics

**Retrieval Quality**:
```
Precision@k = (relevant docs in top-k) / k
Recall@k = (relevant docs in top-k) / (total relevant docs)
MRR = 1 / (rank of first relevant doc)
NDCG@k = Normalized Discounted Cumulative Gain
```

**Generation Quality**:
- ROUGE (recall-oriented overlap)
- BLEU (precision-oriented overlap)
- BERTScore (semantic similarity)

---

## References

### Models
- **SentenceTransformers**: [https://www.sbert.net/](https://www.sbert.net/)
- **all-MiniLM-L6-v2**: [HuggingFace Model Card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **LLaMA 3.1**: [Meta AI Research](https://ai.meta.com/llama/)

### Papers
- **Attention Is All You Need** (Vaswani et al., 2017): Transformer architecture
- **BERT** (Devlin et al., 2019): Bidirectional encoder representations
- **Sentence-BERT** (Reimers & Gurevych, 2019): Sentence embeddings using siamese networks
- **RAG** (Lewis et al., 2020): Retrieval-Augmented Generation for Knowledge-Intensive NLP

### Mathematical Foundations
- **Linear Algebra**: Vector spaces, norms, inner products
- **Information Retrieval**: Similarity measures, ranking
- **Deep Learning**: Transformers, attention mechanisms

---

## License

This project is for educational purposes, demonstrating RAG fundamentals with production-grade models in a clean, minimal implementation.
