import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.getenv("MODEL_DIR") or os.path.join(ROOT_DIR, 'weights')

BASE_URL = 'https://github.com/beasteers/EgoHOS/releases/download/v1/{}'
FILES = ['seg_twohands_ccda', 'twohands_to_cb_ccda', 'twohands_cb_to_obj1_ccda', 'twohands_cb_to_obj2_ccda']

def ensure_checkpoint(path=None):
    path = path or os.path.join(MODEL_DIR, 'egohos')
    os.makedirs(os.path.dirname(path), exist_ok=True)

    for name in FILES:
        pathi = os.path.join(path, name)
        if not os.path.isdir(pathi):
            print("No checkpoint found. Downloading...")
            def show_progress(i, size, total):
                print(f'downloading checkpoint to {path}: {i * size / total:.2%}', end="\r")
            
            import urllib.request, zipfile
            out_path = f'{name}.zip'
            url = BASE_URL.format(f'{name}.zip')
            urllib.request.urlretrieve(url, out_path, show_progress)
            try:
                with zipfile.ZipFile(out_path, 'r') as zf:
                    zf.extractall(pathi)
            except zipfile.BadZipFile:
                print(url)
                print(open(out_path).read(200))
                raise
            finally:
                os.remove(out_path)
    return path
