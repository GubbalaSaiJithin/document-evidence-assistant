# Document retrieval

TF-IDF weights terms by their frequency in a document and rarity across the collection. Cosine similarity compares normalized document and query vectors. This lexical approach can miss paraphrases and synonyms. Dense embeddings can capture semantic similarity, but their usefulness should be evaluated on representative queries.

Chunking splits longer documents into smaller passages. Overlap may preserve context across boundaries but can return repetitive passages. Keep document names, page numbers and chunk identifiers with each passage. Evaluate whether the expected source appears among the top results and measure reciprocal rank. Similarity scores are not probabilities that a factual answer is correct.
