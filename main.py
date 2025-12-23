import argparse
import sys
from core.paper_ops import add_paper, search_paper, batch_organize
from core.image_ops import search_image, index_images

def main():
    parser = argparse.ArgumentParser(description="本地 AI 智能文献与图像管理助手")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 1. 添加论文命令
    add_parser = subparsers.add_parser("add_paper", help="添加并自动分类单篇论文")
    add_parser.add_argument("path", type=str, help="PDF 文件路径")
    add_parser.add_argument("--topics", type=str, required=True, help="分类主题列表，逗号分隔 (例如: 'CV,NLP,RL')")

    # 2. 搜索论文命令
    search_p_parser = subparsers.add_parser("search_paper", help="语义搜索论文")
    search_p_parser.add_argument("query", type=str, help="搜索关键词或问题")
    search_p_parser.add_argument("--index-only", action="store_true", help="仅返回文件路径列表 (文件索引模式)")
    search_p_parser.add_argument("-t", "--threshold", type=float, default=0.4, help="相似度阈值 (默认 0.4)")

    # 3. 批量整理命令
    batch_parser = subparsers.add_parser("batch_organize", help="批量整理文件夹中的 PDF")
    batch_parser.add_argument("folder", type=str, help="包含混乱 PDF 的文件夹路径")
    batch_parser.add_argument("--topics", type=str, required=True, help="分类主题列表")

    # 4. 索引图片命令 (新增，用于初始化图片库)
    idx_img_parser = subparsers.add_parser("index_images", help="扫描并索引图片")
    idx_img_parser.add_argument("--source", type=str, help="图片源文件夹 (可选)，不填则扫描 data/images")

    # 5. 以文搜图命令
    search_i_parser = subparsers.add_parser("search_image", help="以文搜图")
    search_i_parser.add_argument("query", type=str, help="图片描述")
    search_i_parser.add_argument("--index-only", action="store_true", help="仅返回文件路径")
    search_i_parser.add_argument("-t", "--threshold", type=float, default=0.25, help="相似度阈值 (默认 0.25)")
    search_i_parser.add_argument("-n", type=int, default=3, help="检索候选数量 (默认 5)")

    args = parser.parse_args()

    if args.command == "add_paper":
        add_paper(args.path, args.topics)
    elif args.command == "search_paper":
        search_paper(args.query, simple_list=args.index_only, threshold=args.threshold)
    elif args.command == "batch_organize":
        batch_organize(args.folder, args.topics)
    elif args.command == "index_images":
        index_images(args.source)
    elif args.command == "search_image":
        search_image(args.query,
                     n_results=args.n,
                     threshold=args.threshold,
                     simple_list=args.index_only)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()