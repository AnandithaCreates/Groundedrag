"""
Turns files in a directory into Chunk objects for the vector store.

Chunking strategy: split on blank-line paragraph breaks, then greedily pack
paragraphs into ~800 character chunks with a small overlap so an idea that
spans a paragraph boundary isn't cut in half. Simple on purpose -- this is
the piece you'd swap for token-aware or heading-aware splitting if your
docs demanded it, and you should be ready to explain why this one was
good enough for your corpus.
"""
import glob
import os
from typing import List

from app.vector_store import Chunk

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def _read_files(data_dir: str) -> List[tuple]:
    paths = sorted(glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True))
    docs = []
    for path in paths:
        if not path.lower().endswith((".md", ".txt")):
            continue
        with open(path, encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read()))
    return docs


def _chunk_text(text: str) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= CHUNK_SIZE:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            # start next chunk with a small overlap from the tail of the previous one
            overlap = current[-CHUNK_OVERLAP:] if current else ""
            current = f"{overlap}\n\n{para}".strip()
    if current:
        chunks.append(current)
    return chunks


def load_and_chunk(data_dir: str) -> List[Chunk]:
    docs = _read_files(data_dir)
    if not docs:
        raise ValueError(
            f"No .md or .txt files found in {data_dir}. Add your own documents there."
        )
    chunks = []
    for source, text in docs:
        for i, piece in enumerate(_chunk_text(text)):
            chunks.append(Chunk(id=f"{source}::{i}", text=piece, source=source))
    return chunks
