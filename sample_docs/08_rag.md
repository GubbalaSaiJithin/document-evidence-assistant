# Retrieval-augmented generation

Retrieval-augmented generation, or RAG, retrieves passages and supplies them as context to a language model that generates an answer. Returning search results alone is retrieval, not RAG. Source citations help a reader inspect evidence, but citations alone do not prove that an answer is supported.

Evaluate retrieval quality separately from answer quality. Test questions that have no answer in the collection, verify that cited passages support claims, and record latency. Treat instructions inside documents as document content rather than commands. A similarity threshold may reject irrelevant queries but cannot reliably determine whether a passage answers every part of a question.
