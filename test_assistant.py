import unittest
import tempfile
import json
from pathlib import Path
from assistant import SearchIndex, chunks
from app import create_app

class SearchTests(unittest.TestCase):
    def test_sources_and_no_answer(self):
        index = SearchIndex([{'source': 'sql.txt', 'page': None, 'chunk': 1, 'text': 'SQL joins combine rows from database tables.'},
                             {'source': 'ml.txt', 'page': 2, 'chunk': 1, 'text': 'Train models with separate validation data.'}])
        self.assertEqual(index.search('SQL joins')[0]['source'], 'sql.txt')
        self.assertEqual(index.search('volcano astronomy'), [])
        with self.assertRaises(ValueError):
            index.search(' ')

    def test_chunk_overlap_and_bounds(self):
        self.assertEqual(chunks('one two three four five six', size=4, overlap=1), ['one two three four', 'four five six'])
        with self.assertRaises(ValueError):
            chunks('text', size=2, overlap=2)

    def test_http_search_and_validation(self):
        client = create_app().test_client()
        self.assertEqual(client.get('/').status_code, 200)
        response = client.post('/api/search', json={'question': 'How can I prevent data leakage?'})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['results'])
        self.assertEqual(client.post('/api/search', json={'question': ' '}).status_code, 400)
        self.assertEqual(client.post('/api/search', json=['invalid']).status_code, 400)

    def test_short_definition_query_retrieves_relevant_source(self):
        from assistant import ROOT, load_documents
        index = SearchIndex(load_documents(ROOT / 'sample_docs'))
        self.assertEqual(index.search('What does MAE measure?')[0]['source'], '03_forecasting.md')

    def test_document_folder_skips_directories_with_text_extensions(self):
        from assistant import load_documents
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'folder.md').mkdir()
            (root/'doc.txt').write_text('SQL queries read database tables.',encoding='utf-8')
            self.assertEqual(len(load_documents(root)),1)

    def test_mrr_uses_actual_chunk_rank(self):
        from assistant import evaluate
        class FakeIndex:
            def answer(self,query):
                return {'results':[{'source':'wrong.md'},{'source':'wrong.md'},{'source':'right.md'}], 'latency_ms':1.}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'questions.json'
            path.write_text(json.dumps([{'question':'q','expected_source':'right.md'}, {'question':'other','expected_source':None}]))
            self.assertAlmostEqual(evaluate(FakeIndex(),path)['source_mrr_at_3_chunks'],1/3)

if __name__ == '__main__':
    unittest.main()
