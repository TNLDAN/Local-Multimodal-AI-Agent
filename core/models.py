from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import torch

class ModelLoader:
    _text_model = None
    _clip_model = None
    _clip_processor = None
    _clip_tokenizer = None

    @classmethod
    def get_text_model(cls):
        """加载 SentenceTransformer 用于文本/论文嵌入"""
        if cls._text_model is None:
            print("正在加载文本模型 (all-MiniLM-L6-v2)...")
            cls._text_model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._text_model

    @classmethod
    def get_clip_model(cls):
        """加载 CLIP 用于图像嵌入"""
        if cls._clip_model is None:
            print("正在加载 CLIP 模型 (openai/clip-vit-base-patch32)...")
            cls._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            cls._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            cls._clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        return cls._clip_model, cls._clip_processor, cls._clip_tokenizer

def get_text_embedding(text):
    model = ModelLoader.get_text_model()
    return model.encode(text).tolist()

def get_image_embedding(image):
    model, processor, _ = ModelLoader.get_clip_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.squeeze().tolist()

def get_clip_text_embedding(text):
    """用于以文搜图的文本嵌入 (使用 CLIP 的文本编码器)"""
    model, _, tokenizer = ModelLoader.get_clip_model()
    inputs = tokenizer([text], padding=True, return_tensors="pt")
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features.squeeze().tolist()