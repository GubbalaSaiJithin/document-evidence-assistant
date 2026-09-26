# Document Evidence Assistant

A local document-search project for NLP and AI application internships. It indexes text, Markdown and text-based PDFs and returns matching excerpts with filename, page number (for PDFs), chunk ID and similarity score. It includes a Flask web interface and JSON API.

**Current scope:** word/character TF-IDF retrieval with a Flask search interface, plus an optional CLI RAG pipeline using a local FLAN-T5-small model. Both modes have been run locally. Generation is a small-model baseline with documented failures, not a reliable production assistant. Haystack, dense embeddings and vector databases are not implemented here.

## Run on Windows PowerShell

Open a terminal in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe assistant.py --query "How do I prevent data leakage?"
.\.venv\Scripts\python.exe assistant.py --evaluate
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5050 in a browser. Stop with Ctrl+C. No API keys or GPU are required. This is a local development server.

Add documents to `sample_docs` and restart the app to rebuild the index. To query a different directory from the CLI:

```powershell
.\.venv\Scripts\python.exe assistant.py --docs .\private_docs --query "Your question"
```

`private_docs` is ignored by Git. Do not put personal documents in the public sample collection. Scanned PDFs need an OCR extension; OCR is not included. Only immediate files in the selected folder are indexed.

## How it works

Files → extract text per page → overlapping 140-word chunks → word TF-IDF unigrams/bigrams and character 3–5-grams → weighted cosine similarity (65% word, 35% character) → top three excerpts above a fixed 0.05 threshold. The character component was added after a short MAE definition query exposed a word-matching failure. The threshold and weights are development choices, not parameters selected on an independent benchmark.

The threshold is a simple heuristic, not a calibrated answerability score. Source text is displayed as text in the search UI. Indexing occurs once on startup; queries reuse the fitted vectorizers. The separate RAG CLI sends the top two passages to the local model; it does not execute tools or document instructions, but prompt text alone is not a reliable protection against misleading document content.

## Run the optional local RAG pipeline

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-rag.txt
.\.venv\Scripts\python.exe download_model.py
.\.venv\Scripts\python.exe rag.py --query "What does INNER JOIN return?"
.\.venv\Scripts\python.exe rag.py --demo
```

The model download needs internet and about 300 MB of storage; PyTorch and the other libraries need additional disk space. Model inference runs on CPU with no API key, and uses only local files after download. It may take several seconds to load. `model_revision.json` records the downloaded revision. Local model weights are available in the original workspace but excluded from the delivery ZIP and Git.

[Google FLAN-T5-small](https://huggingface.co/google/flan-t5-small) is Apache-2.0 licensed. The model is selected for a modest local demonstration, not because it represents current state-of-the-art answer quality. The Flask page is the retrieval interface; generation is currently available through `rag.py` only. Adding a generation UI/API is a useful extension.

The generator uses up to two passages, a 512-token input limit and deterministic decoding. It returns the generated answer alongside retrieved source metadata. These are sources to inspect, not automatically verified claim-level citations. No-result queries skip generation. A conservative lexical evidence-coverage check now also rejects questions with too many unmatched content terms. This fixes the documented password-query failure, but is not semantic entailment: it can reject useful paraphrases and can still miss unsupported claims.

## What was verified locally

- CLI retrieval and evaluation on eight original, short sample documents.
- Flask homepage, valid search request, empty input and invalid JSON payload using the Flask test client.
- Chunk overlap, source metadata and an irrelevant-query case.
- Text extraction and page numbering on a generated two-page PDF fixture.
- Nine retrieval/RAG regression checks, including real local model inference, unsupported-question abstention, directories with text-file suffixes and correct reciprocal rank when a document has multiple chunks.
- A live local Flask server and HTTP search request.

The authored 16-question smoke set has 12 answerable and four unanswerable questions. Results: expected source present in the top three chunks for 12/12 answerable questions, first for 11/12 (MRR 0.958); abstention for only 2/4 unanswerable questions. These examples are deliberately small and related to the sample documents. They are not evidence of general search quality or a CV-worthy 100% accuracy claim. Two overlapping-vocabulary negatives expose a real limitation: retrieving something similar does not mean the answer exists.

See `evaluation_results.json` for retrieval results and `generation_examples.json` for four actual local generation examples. The retrieval evaluation does not measure generated-answer correctness. `GENERATION_REVIEW.md` records manual observations. The browser's JavaScript has not been automated end-to-end; server routes and HTML responses were checked.

## Build the portfolio version

1. Use 20–50 permitted public technical documents from a coherent domain and record source URLs and licenses.
2. Write at least 40 realistic questions with expected source passages, including paraphrases and unanswerable cases. Separate development and final evaluation questions before tuning. The included negative examples helped develop the abstention rule, so they are regression checks, not independent evidence of generalisation.
3. Compare this TF-IDF baseline with BM25 and dense embeddings. Measure source hit rate/recall@k and MRR, and inspect failures. For multiple relevant sources, implement proper recall using the full set of relevant sources.
4. Compare the small FLAN-T5 baseline with a more capable local model through Haystack/Ollama. Improve answer grounding and the no-answer policy, and add a generation UI/API. Check the chosen model's separate license and hardware requirements.
5. Manually evaluate whether each answer is correct and each citation supports its claim; measure latency. Include tests for misleading instructions in retrieved documents.

Estimated effort: 10–15 focused hours to understand and improve retrieval; roughly 15–25 more to extend and evaluate generation. Hardware and prior knowledge can change this substantially. The included Transformers RAG implementation has run locally. The Haystack/Ollama migration has not been implemented or tested.

## Official open-source starting points for the extension

- [Haystack on GitHub](https://github.com/deepset-ai/haystack), Apache-2.0 licensed.
- [Official first RAG pipeline tutorial](https://haystack.deepset.ai/tutorials/27_first_rag_pipeline).
- [Official Ollama integration and examples](https://github.com/deepset-ai/haystack-integrations/blob/main/integrations/ollama.md).

Check the tutorial and installed framework version together; do not mix version-specific examples. Some tutorials use hosted providers and require keys/charges. The included retrieval starter requires neither.

## Interview questions

- Why can lexical search fail on paraphrases?
- How do chunk size and overlap affect retrieval?
- Why is cosine similarity not a probability of correctness?
- What happens when the question has no answer but shares keywords with a document?
- What do source hit rate and reciprocal rank measure?
- How would you evaluate retrieval separately from generated answers?
- What did you implement beyond the provided starter?

## CV wording after your own work

For the retrieval version: “Built a local document-search application with PDF page citations, TF-IDF retrieval and a Flask API; evaluated source retrieval and out-of-scope queries on a documented question set.”

After reproducing and improving the RAG pipeline: “Extended a local document-Q&A pipeline with retrieval-augmented generation, source metadata and separate retrieval/answer evaluation; analysed unsupported answers and out-of-scope queries.” Describe your actual changes and measured results. Do not claim that using a pretrained model is training or fine-tuning it.

Code and sample documents were authored with AI assistance for this starter and are provided under the included MIT license. Dependencies retain their own licenses. The sample notes are original explanatory text, not copied documentation.
