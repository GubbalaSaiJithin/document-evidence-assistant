from pathlib import Path
from flask import Flask, jsonify, request, render_template_string
from assistant import SearchIndex, load_documents

ROOT = Path(__file__).resolve().parent

def create_app(docs=ROOT / 'sample_docs'):
    app = Flask(__name__)
    index = SearchIndex(load_documents(docs))

    @app.get('/')
    def home():
        return render_template_string('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Document Evidence Assistant</title>
        <style>body{font:17px system-ui;max-width:850px;margin:50px auto;padding:20px;color:#173047}input{width:75%;padding:12px}button{padding:12px}article{background:#f1f5f8;padding:18px;margin:15px 0;border-radius:8px}small{color:#555}</style>
        <h1>Document Evidence Assistant</h1><p>Search a local collection and read the matching source excerpts.</p>
        <form id="form"><input id="q" aria-label="Question" placeholder="How do I prevent data leakage?" required maxlength="2000"><button>Search</button></form>
        <p id="status" role="status"></p><div id="results"></div><p><small>TF-IDF retrieval baseline. Scores measure text similarity, not factual confidence. No LLM is used.</small></p>
        <script>document.getElementById('form').onsubmit=async(e)=>{e.preventDefault();const box=document.getElementById('results');box.replaceChildren();const status=document.getElementById('status');status.textContent='Searching...';try{const r=await fetch('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:document.getElementById('q').value})});const data=await r.json();status.textContent=data.message||data.error;for(const hit of data.results||[]){const a=document.createElement('article');const h=document.createElement('h3');h.textContent=hit.source+(hit.page?' · page '+hit.page:'')+' · chunk '+hit.chunk;const p=document.createElement('p');p.textContent=hit.text;const s=document.createElement('small');s.textContent='Similarity: '+hit.score.toFixed(3);a.append(h,p,s);box.append(a)}}catch(err){status.textContent='Could not connect. Check that the local server is running.'}};</script></html>''')

    @app.post('/api/search')
    def search():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get('question'), str):
            return jsonify(error='Provide a JSON object with a question string.'), 400
        question = payload['question']
        if not question.strip() or len(question) > 2000:
            return jsonify(error='Question must contain 1-2000 characters.'), 400
        return jsonify(index.answer(question))

    return app

if __name__ == '__main__':
    create_app().run(host='127.0.0.1', port=5050, debug=False)
