from pathlib import Path


MODULE_DIR = Path(file).absolute().parent
BASE_DIR = MODULE_DIR.parent
DATA_DIR = BASE_DIR / "data"

