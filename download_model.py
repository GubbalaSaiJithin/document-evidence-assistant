"""Download only the public model assets into this project, never private documents."""
from pathlib import Path
import os
import json

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.hf_cache'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

if __name__ == '__main__':
    from huggingface_hub import snapshot_download
    revision = '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab'
    folder = ROOT / 'models/flan-t5-small'
    snapshot_download(repo_id='google/flan-t5-small', revision=revision, local_dir=str(folder),
        allow_patterns=['config.json', 'generation_config.json', 'model.safetensors',
                        'tokenizer_config.json', 'special_tokens_map.json', 'tokenizer.json', 'spiece.model', 'README.md'])
    (ROOT / 'model_revision.json').write_text(json.dumps({'model': 'google/flan-t5-small', 'revision': revision}, indent=2), encoding='utf-8')
    print('Model downloaded to', folder)
