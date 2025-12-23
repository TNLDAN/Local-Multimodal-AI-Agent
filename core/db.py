import chromadb
from chromadb.config import Settings
from .config import DB_DIR


class VectorDB:
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            cls._client = chromadb.PersistentClient(path=DB_DIR)
        return cls._client

    @classmethod
    def get_collection(cls, name):
        client = cls.get_client()
        # 针对论文使用余弦相似度，图片使用余弦相似度
        return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})