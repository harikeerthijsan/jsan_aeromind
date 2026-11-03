"""Point every test run at a throwaway database and upload folder.

Without this, `pytest` started in backend/ used the real tower_vision.db and uploads/ (the app reads DATABASE_URL and
UPLOAD_DIR at import time), leaving test inspections and files in the working data. pytest imports conftest.py before
tests.py, so these settings are in place before the app is imported. Set AEROMIND_TEST_KEEP=1 to keep the folder.
"""
import os, shutil, tempfile, atexit
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix='aeromind-tests-'))
os.environ['DATABASE_URL'] = f'sqlite:///{(_tmp / "test.db").as_posix()}'
os.environ['UPLOAD_DIR'] = str(_tmp / 'uploads')
os.environ['VENDOR_REFERENCE_DIR'] = str(_tmp / 'vendor_refs')
os.environ['RECON_DIR'] = str(_tmp / 'reconstructions')
os.environ['COLMAP_PATH'] = 'off'  # tests stub the photogrammetry step; don't pick up an installed COLMAP
os.environ['AEROMIND_SKIP_DOTENV'] = '1'  # tests never read the developer's backend/.env
os.environ['YOLO_MODEL_PATH'] = ''  # tests stub the detector; never load real weights
os.environ.setdefault('ENABLE_DEMO_SEED', 'true')  # tests rely on demo site 1
if not os.getenv('AEROMIND_TEST_KEEP'):
    atexit.register(shutil.rmtree, _tmp, True)
