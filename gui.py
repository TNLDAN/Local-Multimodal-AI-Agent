import streamlit as st
import os
import subprocess
import platform
from core.paper_ops import add_paper, batch_organize
from core.image_ops import index_images
from core.db import VectorDB
from core.models import ModelLoader, get_clip_text_embedding


# --- 页面配置 ---
st.set_page_config(page_title="本地 AI 智能助手", layout="wide", page_icon="🤖")

st.title("🤖 本地 AI 智能文献与图像管理助手")

# --- 侧边栏：功能选择 ---
st.sidebar.header("功能导航")
# 使用 radio 组件，所有选项直接显示
menu = st.sidebar.radio(
    "请选择功能:",
    ["➕ 添加单篇文献", "📂 批量整理文献", "📄 语义搜文献", "🔄 更新图片索引", "🖼️ 以文搜图"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("提示: \n使用左侧导航栏切换不同功能模块。")


# --- 搜索逻辑函数 ---
def st_search_paper(query, threshold):
    collection = VectorDB.get_collection("papers")
    model = ModelLoader.get_text_model()
    query_emb = model.encode(query).tolist()

    results = collection.query(query_embeddings=[query_emb], n_results=10)

    if not results['documents'] or not results['documents'][0]:
        st.warning("未找到任何内容。")
        return

    found_count = 0
    for i, doc in enumerate(results['documents'][0]):
        dist = results['distances'][0][i]
        similarity = 1 - dist

        if similarity >= threshold:
            found_count += 1
            meta = results['metadatas'][0][i]

            # 使用 expander 显示
            with st.expander(f"[{found_count}] {meta['filename']} (相似度: {similarity:.4f})"):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**分类**: `{meta['category']}` | **页码**: `{meta['page']}`")
                    st.info(doc)  # 显示完整片段
                    st.text(f"路径: {meta['path']}")

    if found_count == 0:
        st.warning(f"没有找到相似度 > {threshold} 的结果。")


def st_search_image(query, threshold, n_results=10):
    collection = VectorDB.get_collection("images")
    query_emb = get_clip_text_embedding(query)
    results = collection.query(query_embeddings=[query_emb], n_results=n_results)

    if not results['ids'] or not results['ids'][0]:
        st.warning("未找到相关图片。")
        return

    cols = st.columns(3)
    count = 0
    for i, _ in enumerate(results['ids'][0]):
        dist = results['distances'][0][i]
        similarity = 1 - dist

        if similarity >= threshold:
            meta = results['metadatas'][0][i]
            with cols[count % 3]:
                if os.path.exists(meta['path']):
                    st.image(meta['path'], caption=f"相似度: {similarity:.4f}")
                    st.caption(meta['filename'])

                else:
                    st.error("图片文件丢失")
            count += 1

    if count == 0:
        st.warning(f"没有找到相似度 > {threshold} 的图片。")


# --- 主界面逻辑 ---

if menu == "📄 语义搜文献":
    st.header("🔍 语义搜索文献")
    with st.form("search_form"):
        c1, c2 = st.columns([4, 1])
        with c1:
            query = st.text_input("请输入问题或关键词")
        with c2:
            threshold = st.slider("相似度阈值", 0.0, 1.0, 0.4, 0.05)
        submitted = st.form_submit_button("🔍 开始搜索")

    if submitted and query:
        with st.spinner("正在搜索知识库..."):
            st_search_paper(query, threshold)

elif menu == "🖼️ 以文搜图":
    st.header("🎨 以文搜图")
    with st.form("img_form"):
        c1, c2 = st.columns([4, 1])
        with c1:
            query = st.text_input("请输入图片描述")
        with c2:
            threshold = st.slider("相似度阈值", 0.0, 1.0, 0.25, 0.05)
        submitted = st.form_submit_button("🖼️ 搜索图片")

    if submitted and query:
        with st.spinner("正在分析图片库..."):
            st_search_image(query, threshold)

elif menu == "➕ 添加单篇文献":
    st.header("📤 添加单篇文献")
    uploaded_file = st.file_uploader("上传 PDF 文件", type="pdf")
    topics_str = st.text_input("分类主题 (逗号分隔)", value="NLP, CV, RL, RecSys")

    if st.button("处理并归档") and uploaded_file:
        with st.spinner("正在读取、分类并建立索引..."):
            os.makedirs("data", exist_ok=True)
            temp_path = os.path.join("data", uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            add_paper(temp_path, topics_str)
            st.success(f"成功处理: {uploaded_file.name}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

elif menu == "📂 批量整理文献":
    st.header("📚 批量整理文件夹")
    folder_path = st.text_input("请输入待整理的文件夹绝对路径")
    topics_str = st.text_input("分类主题", value="Collaborative Filtering, Deep Learning, Graph Neural Networks")

    if st.button("开始整理") and folder_path:
        if os.path.exists(folder_path):
            with st.spinner("正在批量扫描和处理..."):
                batch_organize(folder_path, topics_str)
                st.success("批量整理完成！")
        else:
            st.error("路径不存在，请检查。")

elif menu == "🔄 更新图片索引":
    st.header("🖼️ 更新图片索引")
    source_dir = st.text_input("图片源文件夹")

    if st.button("开始建立索引"):
        with st.spinner("正在扫描图片并计算向量..."):
            index_images(source_dir if source_dir else None)
            st.success("索引更新完毕！")

# --- 页脚 ---
st.markdown("---")
st.caption("Local AI Agent | Powered by Sentence-Transformers & CLIP & ChromaDB")