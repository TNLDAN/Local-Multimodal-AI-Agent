import os

# 基础路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 子目录
PAPER_DIR = os.path.join(DATA_DIR, "papers")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
DB_DIR = os.path.join(DATA_DIR, "chromadb")

# 确保目录存在
os.makedirs(PAPER_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)