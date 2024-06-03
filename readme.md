# Search Engine Project

This project is developed as an assignment for the Information Retrieval subject. It illustrates how search engines can be implemented using different techniques, including TF-IDF based vectorization and embedding-based vectorization.

## Techniques Used

### 1. TF-IDF Based Vectorization
### 2. Embedding Based Vectorization

| **TF-IDF Based Flow** | **Embedding Based Flow** |
|-----------------------|--------------------------|
| 1. User sends the request. <br> 2. Load the stored TF-IDF model. <br> 3. Convert the query to TF-IDF vector. <br> 4. Load the TF-IDF matrix. <br> 5. Apply cosine similarity between query vector and matrix. <br> 6. Get the top k matching results and return them to the user. | 1. User sends the request. <br> 2. Load the stored word embedding model. <br> 3. Convert the query to vector embedding. <br> 4. Open connection with vector DB. <br> 5. Query vector DB for similar docs. <br> 6. Get the top k matching results and return them to the user. |


## Used Libraries

- **Numpy**
- **Chroma DB** (Vector DB)
- **Gensim** (for Word2Vec)
- **Sklearn**
- **FastAPI**
- **NLTK** (for text processing)

## Contributors

- Alaa Aldeen Zamel
- Anas Rish
- Anas Durra
- Mohammed Hadi Barakat
- Mohammed Fares Dabbas

## Datasets Used

- **Antique**: [Link](https://ir-datasets.com/antique.html#antique/train)
- **Wikipedia**: [Link](https://ir-datasets.com/wikir.html#wikir/en1k/training)

This project demonstrates the fundamental steps involved in building a search engine using two distinct approaches: one based on TF-IDF vectorization and the other on word embeddings. By employing these techniques, we aim to provide an efficient and accurate retrieval of relevant documents in response to user queries.