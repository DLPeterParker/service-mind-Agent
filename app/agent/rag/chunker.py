"""按 Markdown 二级标题切分知识库文档。

切分策略：
- 以 `## ` 二级标题为切分边界，每个二级章节作为一个 chunk。
- 二级章节内的三级/四级小节保留在同一 chunk 中（保持语义完整）。
- 文档开头到第一个二级标题之间的内容（含一级标题和导语）作为 "概览" chunk。
- chunk 长度超过 ~1200 字时按段落进一步切分，避免单个 chunk 过长。

每个 chunk 保留：
- doc：文档名（不含扩展名），如 "退换货政策"
- section：章节标题，如 "二、质量问题退换货"
- text：chunk 全文（含小节结构）
- chunk_id：稳定的字符串 id，便于增量更新
"""

import re
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_CHUNK_CHARS = 1200


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    section: str
    text: str

    # 把 Chunk 数据类对象，转换成键值对形式的普通字典
    def to_dict(self) -> dict:
        return asdict(self)


def chunk_markdown_dir(kb_dir: Path) -> list[Chunk]:
    """扫描目录下所有 .md 文件，逐一切分并汇总。"""
    chunks: list[Chunk] = []
    for md_path in sorted(kb_dir.glob("*.md")):
        chunks.extend(_chunk_one_file(md_path))
    return chunks


def _chunk_one_file(md_path: Path) -> list[Chunk]:
    raw = md_path.read_text(encoding="utf-8")
    doc_name = md_path.stem
    sections = _split_by_h2(raw)

    out: list[Chunk] = []
    for idx, (section_title, section_body) in enumerate(sections):
        text = section_body.strip()
        if not text:
            continue

        if len(text) <= MAX_CHUNK_CHARS:
            out.append(_make_chunk(doc_name, section_title, text, idx, 0))
            continue
        # 内容太长， 按段落分割
        for sub_idx, piece in enumerate(_split_long(text)):
            out.append(_make_chunk(doc_name, section_title, piece, idx, sub_idx))
    return out


def _split_by_h2(raw: str) -> list[tuple[str, str]]:
    """返回 [(section_title, section_body), ...]。

    第一个 section 是文档头部（H1 + 导语），title 取 H1 文本。
    """
    lines = raw.splitlines()  # 将原始文本 按行分割
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_body: list[str] = []

    for line in lines:
        if line.startswith("# ") and not current_body and not sections:
            current_title = line[2:].strip() + " · 概览"  # 删除"# "前缀,取一级标题
            continue  # 跳过当前循环的剩余代码，直接去抓下一行文本
        if line.startswith("## "):
            if current_body:
                sections.append((current_title or "概览", current_body))
            current_title = line[3:].strip()
            current_body = []
            continue
        current_body.append(line)  # 装入普通正文

    if current_body:  # 最后部分的正文添加标题，并且添加到 sections 中 收起来
        sections.append((current_title or "概览", current_body))

    # 把「标题 + 正文行列表」的格式，转换成「标题 + 完整正文字符串」的格式
    return [(t, "\n".join(b).strip()) for t, b in sections]


def _split_long(text: str) -> list[str]:
    """按段落贪心打包到 MAX_CHUNK_CHARS。"""
    paragraphs = re.split(
        r"\n\s*\n", text
    )  # 按段落分割，用正则识别“一个或多个空白字符的换行” 提取自然段落
    pieces: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
            # 当前缓冲区不为空，且加上新段落会超过限制
        if buf and buf_len + len(p) + 2 > MAX_CHUNK_CHARS:
            pieces.append("\n\n".join(buf))
            buf = [p]
            buf_len = len(p)
        else:
            buf.append(p)
            buf_len += len(p) + 2  # +2 是因为段落间有两个换行符 "\n\n"
    if buf:
        pieces.append("\n\n".join(buf))
    return pieces


def _make_chunk(doc: str, section: str, text: str, idx: int, sub_idx: int) -> Chunk:
    """创建一个标准化的文本块（Chunk）对象。
    将文档名、章节标题、文本内容等信息整合为一个结构化的 Chunk 对象，
    便于后续向量存储和检索。
    """
    # 生成唯一ID：格式为「文档名#章节序号-子块序号」，序号补零到两位
    chunk_id = f"{doc}#{idx:02d}-{sub_idx:02d}"

    # 构建正文：在内容前添加「文档名 · 章节名」的标识头
    body = f"【{doc} · {section}】\n{text}"

    # 创建并返回 Chunk 对象
    return Chunk(chunk_id=chunk_id, doc=doc, section=section, text=body)
