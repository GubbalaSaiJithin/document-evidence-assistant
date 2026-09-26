"""Offline document retrieval with source excerpts. This baseline uses no LLM."""
from pathlib import Path
import argparse
import json
import re
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent

def chunks(text, size=140, overlap=30):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError('Require size > 0 and 0 <= overlap < size.')
    words = text.split()
    result = []
    for start in range(0, len(words), size - overlap):
        result.append(' '.join(words[start:start + size]))
        if start + size >= len(words):
            break
    return result

def load_documents(folder):
    documents = []
    for path in sorted(Path(folder).iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() in {'.txt', '.md'}:
            pages = [(None, path.read_text(encoding='utf-8'))]
        elif path.suffix.lower() == '.pdf':
            from pypdf import PdfReader
            pages = [(number, page.extract_text() or '') for number, page in enumerate(PdfReader(path).pages, 1)]
        else:
            continue
        for page, text in pages:
            for number, content in enumerate(chunks(text), 1):
                if re.search(r'\w{2,}', content):
                    documents.append({'source': path.name, 'page': page, 'chunk': number, 'text': content})
    if not documents:
        raise ValueError('No extractable text found. Add text/Markdown or text-based PDF documents; OCR is not included.')
    return documents

class SearchIndex:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([doc['text'] for doc in documents])
        self.char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True)
        self.char_matrix = self.char_vectorizer.fit_transform([doc['text'] for doc in documents])

    def search(self, query, top_k=3, minimum_score=.05):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('Enter a non-empty question.')
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 20:
            raise ValueError('top_k must be between 1 and 20.')
        if not 0 <= minimum_score <= 1:
            raise ValueError('minimum_score must be between 0 and 1.')
        word_scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        char_scores = cosine_similarity(self.char_vectorizer.transform([query]), self.char_matrix)[0]
        scores = .65 * word_scores + .35 * char_scores
        order = np.argsort(-scores, kind='stable')[:top_k]
        return [{**self.documents[index], 'score': float(scores[index])} for index in order
                if scores[index] >= minimum_score and scores[index] > 0]

    def answer(self, question):
        start = time.perf_counter()
        results = self.search(question)
        return {'question': question, 'mode': 'retrieved excerpts; no generated answer',
                'message': 'Relevant source excerpts:' if results else 'No sufficiently matching passage found. Try different wording.',
                'results': results, 'latency_ms': (time.perf_counter() - start) * 1000}

def evaluate(index, path):
    examples = json.loads(Path(path).read_text(encoding='utf-8'))
    positive, reciprocal, no_answer, latencies, details = [], [], [], [], []
    for example in examples:
        response = index.answer(example['question'])
        # Keep chunk rank: deduplicating filenames would inflate reciprocal rank.
        sources = [hit['source'] for hit in response['results']]
        expected = example['expected_source']
        if expected is None:
            no_answer.append(not sources)
        else:
            positive.append(expected in sources)
            reciprocal.append(1 / (sources.index(expected) + 1) if expected in sources else 0)
        latencies.append(response['latency_ms'])
        details.append({**example, 'retrieved_sources': sources})
    return {'evaluation_type': 'Small authored smoke set; NOT an independent benchmark.',
            'questions': len(examples), 'answerable_questions': len(positive), 'unanswerable_questions': len(no_answer),
            'source_hit_rate_at_3_chunks': float(np.mean(positive)),
            'source_mrr_at_3_chunks': float(np.mean(reciprocal)),
            'unanswerable_abstention_rate': float(np.mean(no_answer)),
            'median_query_ms': float(np.median(latencies)), 'details': details}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs', type=Path, default=ROOT / 'sample_docs')
    parser.add_argument('--query', default='How do I prevent data leakage?')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    index = SearchIndex(load_documents(args.docs))
    result = evaluate(index, ROOT / 'evaluation.json') if args.evaluate else index.answer(args.query)
    if args.evaluate:
        (ROOT / 'evaluation_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
