import unittest
import importlib.util
from pathlib import Path

class RAGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all(importlib.util.find_spec(name) for name in ['torch','transformers']):
            raise unittest.SkipTest('Install requirements-rag.txt to run optional generation tests.')
        if not (Path(__file__).resolve().parent/'models/flan-t5-small/config.json').is_file():
            raise unittest.SkipTest('Run download_model.py to run optional generation tests.')
        from rag import LocalRAG
        cls.rag=LocalRAG()

    def test_known_definition_generates_answer_and_sources(self):
        response=self.rag.answer('What does INNER JOIN return?')
        self.assertTrue(response['generated'])
        self.assertTrue(response['answer'].strip())
        self.assertEqual(response['sources'][0]['source'],'04_sql.md')

    def test_unsupported_password_question_abstains(self):
        response=self.rag.answer('What is the administrator password for the SQL database?')
        self.assertFalse(response['generated'])

    def test_unknown_topic_and_invalid_input(self):
        self.assertFalse(self.rag.answer('Who won the Wimbledon tennis championship?')['generated'])
        for value in ['',None,'a'*501]:
            with self.assertRaises(ValueError):
                self.rag.answer(value)

if __name__=='__main__':
    unittest.main()
