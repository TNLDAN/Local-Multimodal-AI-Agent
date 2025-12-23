import os
from PIL import Image
from .db import VectorDB
from .models import get_image_embedding, get_clip_text_embedding
from .config import IMAGE_DIR
import uuid
import shutil


def index_images(source_dir):
    """扫描目录下的图片并建立索引（带查重功能）"""
    collection = VectorDB.get_collection("images")

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

    # 遍历源目录或默认目录
    scan_dir = source_dir if source_dir else IMAGE_DIR

    count = 0
    skipped = 0

    # 1. 获取数据库中现有的所有图片 ID，用于查重
    existing_ids = set(collection.get()['ids'])

    for root, _, files in os.walk(scan_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_exts:
                # 使用文件名作为唯一 ID（假设同一文件夹下文件名不重复）
                # 为了防止不同文件夹有同名文件，最好加上相对路径，这里简化用文件名
                img_id = file

                # --- 查重逻辑 ---
                if img_id in existing_ids:
                    # print(f"跳过已存在: {file}")
                    skipped += 1
                    continue
                # ----------------

                file_path = os.path.join(root, file)

                if source_dir:
                    target_path = os.path.join(IMAGE_DIR, file)
                    shutil.copy2(file_path, target_path)
                    file_path = target_path

                try:
                    img = Image.open(file_path)
                    embedding = get_image_embedding(img)

                    collection.add(
                        embeddings=[embedding],
                        metadatas=[{"path": file_path, "filename": file}],
                        ids=[img_id]  # 使用文件名作为 ID
                    )
                    count += 1
                    print(f"已索引: {file}")
                except Exception as e:
                    print(f"无法读取图片 {file}: {e}")

    print(f"索引完成。新增: {count} 张，跳过重复: {skipped} 张。")


def search_image(query, n_results=5, threshold=0.2, simple_list=False):
    """
    Args:
        query: 图片描述
        n_results: 候选数量 (建议设大一点，比如 5-10)
        threshold: 相似度阈值 (CLIP 模型对文本-图像的相似度通常较低，建议默认 0.2-0.25)
        simple_list: 仅返回路径列表
    """
    print(f"正在以文搜图: '{query}' (阈值: {threshold}) ...")
    collection = VectorDB.get_collection("images")

    # 使用 CLIP 的文本编码器
    from .models import get_clip_text_embedding
    query_emb = get_clip_text_embedding(query)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=n_results
    )

    if not results['ids'][0]:
        print("未找到任何图片。")
        return

    # 过滤结果
    valid_results = []
    for i, _ in enumerate(results['ids'][0]):
        dist = results['distances'][0][i]
        # ChromaDB 默认 Cosine 距离，相似度 = 1 - 距离
        similarity = 1 - dist

        if similarity >= threshold:
            valid_results.append({
                "meta": results['metadatas'][0][i],
                "score": similarity
            })

    if not valid_results:
        print(f"未找到相似度 > {threshold} 的图片。")
        return

    # --- 模式 1: 仅返回文件列表 ---
    if simple_list:
        print("\n--- 匹配的图片列表 ---")
        for item in valid_results:
            print(item['meta']['path'])
        return

    # --- 模式 2: 详细展示 ---
    print("\n--- 图像搜索结果 ---")
    for i, item in enumerate(valid_results):
        meta = item['meta']
        print(f"[{i + 1}] 相似度: {item['score']:.4f}")
        print(f"    文件: {meta['filename']}")
        print(f"    路径: {meta['path']}")