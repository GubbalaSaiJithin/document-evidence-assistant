# Document Evidence Assistant

Local search across text, Markdown and text-based PDF files, with source excerpts and an optional retrieval-augmented generation pipeline. The Flask interface and JSON API return matching passages. A separate command-line interface uses a local FLAN-T5-small model to generate answers from retrieved context.

## Quick start

From the project directory, using Python 3.10:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe assistant.py --query "How do I prevent data leakage?"
.\.venv\Scripts\python.exe assistant.py --evaluate
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5050. Stop the local development server with Ctrl+C. On macOS/Linux, substitute `.venv/bin/python` for `.\.venv\Scripts\python.exe`.

The eight sample documents and retrieval evaluation questions are included. Add permitted documents to `sample_docs` and restart the app to rebuild the index. For a separate local collection:

```powershell
.\.venv\Scripts\python.exe assistant.py --docs .\private_docs --query "Your question"
```

`private_docs` is ignored by Git. Only immediate files in the selected folder are indexed. Scanned PDFs require OCR, which is not included.

## Retrieval

1. Extract text from each supported file, preserving PDF page numbers.
2. Split each page or text document into 140-word chunks with 30-word overlap.
3. Fit word TF-IDF unigrams/bigrams and character TF-IDF 3-5-grams on the chunks.
4. Transform the question and calculate cosine similarity against the indexed passages.
5. Rank chunks using 65% word similarity and 35% character similarity.
6. Return up to three chunks above a similarity threshold of 0.05, with filename, page, chunk ID and score.

The index is built once at application startup and reused for subsequent queries. The threshold and weights are development choices. Scores describe text similarity, not probabilities of factual correctness. Character matching helps some word-form variations; retrieval still depends heavily on lexical overlap.

`POST /api/search` accepts a JSON object such as `{"question": "What does INNER JOIN return?"}`. The browser renders returned passages as text. Search mode does not generate answers.

## Local answer generation

Install the optional dependencies and download the pinned model revision:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-rag.txt
.\.venv\Scripts\python.exe download_model.py
.\.venv\Scripts\python.exe rag.py --query "What does INNER JOIN return?"
.\.venv\Scripts\python.exe rag.py --demo
.\.venv\Scripts\python.exe -m unittest test_rag -v
```

The model download requires internet and approximately 300 MB of storage; dependencies need additional space. Inference runs on CPU using local files after download, without an API key. Model weights are excluded from Git. `model_revision.json` records the model revision.

The pipeline retrieves up to two passages and checks how many important question terms occur in them. It declines to generate when there are no hits or coverage is below 0.6. Otherwise, it places the available context and question in a prompt with a 512-token input limit and uses deterministic decoding to generate an answer.

Generated output includes the source metadata and the context actually supplied to the model. This uses a pretrained model; no fine-tuning is implemented. Generation is available through `rag.py`; the Flask interface remains a passage-search interface.

## Evaluation

The 16-question development set contains 12 answerable and four unanswerable questions about the sample documents.

| Retrieval measurement | Result |
|---|---:|
| Expected source among the top three chunks | 12/12 answerable questions |
| Expected source ranked first | 11/12 answerable questions |
| Mean reciprocal rank at three chunks | 0.958 |
| No result returned for unanswerable questions | 2/4 questions |

These measurements describe a small development collection, not an independent search benchmark. Queries with overlapping vocabulary can retrieve a passage even when it cannot answer the question. Retrieval evaluation does not measure generated-answer correctness.

`evaluation_results.json` contains per-question retrieval results. `generation_examples.json` contains four actual generation-path examples: two supported factual answers and two refusals to answer unsupported questions. `GENERATION_REVIEW.md` documents the earlier password-query failure and the effect of the lexical coverage check. These cases informed development and are regression examples, not independent evidence of generalisation.

## Verification and limitations

Nine local regression tests passed, covering retrieval, PDF page metadata, chunking, API validation, reciprocal-rank calculation and actual local generation. A live Flask HTTP search also passed. The GitHub workflow runs the six retrieval tests and evaluation without downloading the optional language model. The three generation tests skip when their dependencies or model files are unavailable.

The small model can generate unsupported answers. The coverage check is a lexical heuristic: it can reject useful paraphrases and cannot verify whether each generated claim follows from the context. Returning sources does not establish claim-level citation accuracy. Prompt instructions alone do not reliably handle misleading content inside documents.

Dense embeddings, a vector database, OCR and a generation web interface are not implemented. Browser JavaScript has not been automated end-to-end, and the Flask server has not been deployed publicly.

## Further work

- Evaluate on a larger permitted document collection with separate development and held-out questions.
- Compare lexical retrieval with other retrieval methods using consistent source relevance labels.
- Evaluate generated-answer support, false refusals and latency separately from retrieval quality.
- Add a generation API and interface with explicit source inspection.

## References and licensing

- [Google FLAN-T5-small](https://huggingface.co/google/flan-t5-small), Apache-2.0 licensed, is the pretrained generation model.
- [Haystack](https://github.com/deepset-ai/haystack), its [RAG tutorial](https://haystack.deepset.ai/tutorials/27_first_rag_pipeline), and the [Ollama integration](https://github.com/deepset-ai/haystack-integrations/blob/main/integrations/ollama.md) are references for possible extensions. These frameworks are not used in the current implementation.

The project code and sample documents are provided under the included [MIT license](LICENSE). Model weights and dependencies retain their own licenses.
