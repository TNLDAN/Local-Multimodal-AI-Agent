import os
import shutil
import uuid
import hashlib
from pypdf import PdfReader
from .db import VectorDB
from .models import get_text_embedding, ModelLoader  # 引入 ModelLoader 用于分类
from .config import PAPER_DIR
from sentence_transformers import util  # 用于分类计算


def classify_paper_content(text_summary, topics):
    """
    使用文本摘要进行分类
    """
    text_model = ModelLoader.get_text_model()
    # 只需要前 500 个字符做分类即可
    content_emb = text_model.encode(text_summary[:500])
    topic_embs = text_model.encode(topics)
    scores = util.cos_sim(content_emb, topic_embs)[0]
    best_idx = scores.argmax().item()
    return topics[best_idx]


def extract_text_and_chunk(file_path, chunk_size=500, overlap=50):
    """
    读取 PDF，按页提取，并进行切片。
    返回格式: [{"text": "段落内容...", "page": 1}, ...]
    """
    chunks = []
    try:
        reader = PdfReader(file_path)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue

            # 简单的滑动窗口切片
            text_len = len(text)
            start = 0
            while start < text_len:
                end = start + chunk_size
                # 尽量不在单词中间切断（简单处理：直接切，复杂处理可以找空格）
                chunk_text = text[start:end]

                # 记录该片段的信息
                chunks.append({
                    "text": chunk_text,
                    "page": page_num + 1  # 页码从 1 开始
                })

                # 移动窗口，预留重叠部分
                start += (chunk_size - overlap)

    except Exception as e:
        print(f"解析 PDF 失败: {e}")
        # 如果解析失败，回退到仅使用文件名作为内容（页码为 0）
        chunks.append({"text": os.path.basename(file_path), "page": 0})

    return chunks


def add_paper(file_path, topics_str):
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return

    topics = [t.strip() for t in topics_str.split(',')]
    filename = os.path.basename(file_path)

    print(f"正在处理: {filename} ...")

    # 1. 提取内容并切片
    chunks = extract_text_and_chunk(file_path)

    if not chunks:
        print("警告: 未能从 PDF 提取到任何有效文本，跳过。")
        return

    # 2. 自动分类
    summary_parts = [c['text'] for c in chunks[:3] if isinstance(c['text'], str)]
    summary_text = " ".join(summary_parts) if summary_parts else filename

    category = classify_paper_content(summary_text, topics)
    print(f"自动分类结果: {category}")

    # 3. 移动文件
    target_dir = os.path.join(PAPER_DIR, category)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)

    if os.path.abspath(file_path) != os.path.abspath(target_path):
        shutil.copy2(file_path, target_path)
        print(f"文件已归档至: {target_path}")

    # 4. 存入向量数据库
    collection = VectorDB.get_collection("papers")

    documents = []
    metadatas = []
    ids = []

    valid_chunk_count = 0
    for i, chunk in enumerate(chunks):
        text_content = chunk['text']

        if not isinstance(text_content, str) or not text_content.strip():
            continue

        documents.append(text_content)

        metadatas.append({
            "path": target_path,
            "filename": filename,
            "category": category,
            "page": chunk['page'],
            "chunk_id": i
        })

        # --- 生成确定性 ID ---
        safe_filename = filename.replace(' ', '_').replace('.', '_')
        deterministic_id = f"{safe_filename}_part_{i}"

        ids.append(deterministic_id)
        valid_chunk_count += 1

    if not documents:
        print("错误: 数据清洗后无有效文本片段，无法建立索引。")
        return

    print(f"正在生成向量索引 (共 {valid_chunk_count} 个有效片段)...")

    try:
        model = ModelLoader.get_text_model()
        embeddings = model.encode(documents).tolist()

        # --- 使用 upsert 代替 add ---
        # upsert: 如果 ID 已存在，则更新（覆盖）；如果不存在，则插入。
        collection.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print("索引建立完成 (已去重)。")

    except Exception as e:
        print(f"向量化或入库过程中发生错误: {e}")


def search_paper(query, n_results=5, threshold=0.4, simple_list=False):
    """
    Args:
        query: 搜索问题
        n_results: 最多查询多少个候选（建议设大一点，比如 5 或 10，然后再用阈值过滤）
        threshold: 相似度阈值 (0~1)，低于此值的即使排在前几名也不显示
        simple_list: 是否仅返回文件列表
    """
    collection = VectorDB.get_collection("papers")

    from .models import get_text_embedding
    query_emb = get_text_embedding(query)

    # 先查出 Top N 个候选
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=n_results
    )

    if not results['documents'][0]:
        print("未找到任何内容。")
        return

    # 准备一个列表来暂存符合阈值的结果
    valid_results = []

    # 遍历候选结果
    for i, doc in enumerate(results['documents'][0]):
        dist = results['distances'][0][i]
        similarity = 1 - dist  # 余弦距离转相似度

        # --- 核心逻辑：阈值过滤 ---
        if similarity >= threshold:
            valid_results.append({
                "doc": doc,
                "meta": results['metadatas'][0][i],
                "score": similarity
            })

    # 如果过滤后没有结果
    if not valid_results:
        print(f"未找到相似度 > {threshold} 的相关论文。")
        return

    # --- 模式 1: 文件索引模式 ---
    if simple_list:
        found_files = set()
        for item in valid_results:
            found_files.add(item['meta']['path'])

        print(f"\n--- 匹配的文件列表 (相似度 > {threshold}) ---")
        for path in found_files:
            print(path)
        return

    # --- 模式 2: 详细搜索模式 ---
    print(f"\n--- 语义搜索结果 ('{query}' | 阈值: {threshold}) ---")
    for i, item in enumerate(valid_results):
        meta = item['meta']
        print(f"[{i + 1}] 相似度: {item['score']:.4f}")
        print(f"    来源: {meta['filename']} (第 {meta['page']} 页)")
        print(f"    分类: {meta['category']}")
        print(f"    片段: \"{item['doc'][:150].replace(chr(10), ' ')}...\"")
        print(f"    路径: {meta['path']}\n")

def batch_organize(source_folder, topics_str):
    files = [f for f in os.listdir(source_folder) if f.lower().endswith('.pdf')]
    print(f"发现 {len(files)} 个 PDF 文件，开始整理...")
    for f in files:
        path = os.path.join(source_folder, f)
        add_paper(path, topics_str)