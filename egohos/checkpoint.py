import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.getenv("MODEL_DIR") or os.path.join(ROOT_DIR, 'weights')

BASE_URL = 'https://docs.google.com/uc?export=download&confirm=t&id={}'
FILE_ID = '1LNMQ6TGf1QaCjMgTExPzl7lFFs-yZyqX'

def ensure_checkpoint(path=None):
    path = path or os.path.join(MODEL_DIR, 'egohos')
    if not os.path.isdir(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        print("No checkpoint found. Downloading...")
        def show_progress(i, size, total):
            print(f'downloading checkpoint to {path}: {i * size / total:.2%}', end="\r")
        
        import urllib.request
        urllib.request.urlretrieve(BASE_URL.format(FILE_ID), 'work_dirs.zip', show_progress)
        import zipfile
        with zipfile.ZipFile('work_dirs.zip', 'r') as zf:
            zf.extractall(path)
    return path
