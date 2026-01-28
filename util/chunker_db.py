#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


PG_DSN = "postgresql://postgres:postgres@5.129.220.167:5432/rag_corpus"
PG_SCHEMA = "rag"

IN_ROOT = Path("../clean_data/entries")

@dataclass
class ChunkerConfig:
    target_chars: int = 2300
    max_chars: int = 3000
    min_chars: int = 750

    overlap_max_chars: int = 200
    overlap_max_sentences: int = 1
    overlap_prefix: str = "(контекст) "

    merge_small_callouts: bool = True
    callout_merge_max_chars: int = 260

    allow_inline_atomic_in_buffer: bool = True
    inline_atomic_max_chars: int = 420

    merge_tiny_chunks: bool = True

    ws_re: str = r"[ \t\r\f\v]+"
    nl3_re: str = r"\n{3,}"
    sent_end_re: str = r"(?<=[.!?])\s+"

    md_table_row_re: str = r"^\s*\|.*\|\s*$"
    md_table_min_rows: int = 3
    md_table_scan_lines: int = 10

    code_fence_re: str = r"(^```|^~~~)"
    code_indent_min_ratio: float = 0.35
    code_indent_scan_lines: int = 30
    code_indent_prefixes: tuple[str, ...] = ("    ", "\t")

    max_heading_level: int = 6


# ==============================
# Helpers
# ==============================

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_regexes(cfg: ChunkerConfig) -> dict[str, re.Pattern[str]]:
    return {
        "ws": re.compile(cfg.ws_re),
        "nl3": re.compile(cfg.nl3_re),
        "sent_end": re.compile(cfg.sent_end_re),
        "md_table_row": re.compile(cfg.md_table_row_re, re.MULTILINE),
        "code_fence": re.compile(cfg.code_fence_re, re.MULTILINE),
    }


def norm_text(s: str, rx: dict[str, re.Pattern[str]]) -> str:
    s = (s or "").replace("\xa0", " ")
    s = rx["ws"].sub(" ", s)
    s = rx["nl3"].sub("\n\n", s)
    return s.strip()


def join_nonempty(parts: list[str], rx: dict[str, re.Pattern[str]], sep: str = "\n\n") -> str:
    cleaned = [p for p in (norm_text(x, rx) for x in parts) if p]
    return sep.join(cleaned).strip()


def dedupe_images(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for im in images or []:
        lp = (im.get("local_path") or "").strip()
        if not lp or lp in seen:
            continue
        seen.add(lp)
        out.append({k: v for k, v in im.items() if k in ("local_path", "alt", "caption") and v})
    return out


def block_images(block: dict[str, Any]) -> list[dict[str, Any]]:
    return dedupe_images(block.get("images") or [])


def is_tableish_text(text: str, cfg: ChunkerConfig, rx: dict[str, re.Pattern[str]]) -> bool:
    t = text.strip()
    if not t:
        return False
    lines = t.splitlines()[: cfg.md_table_scan_lines]
    if not lines:
        return False
    hits = sum(1 for ln in lines if rx["md_table_row"].match(ln))
    return hits >= cfg.md_table_min_rows


def is_codeish_text(text: str, cfg: ChunkerConfig, rx: dict[str, re.Pattern[str]]) -> bool:
    t = text.strip()
    if not t:
        return False
    if rx["code_fence"].search(t):
        return True
    lines = t.splitlines()
    scan = lines[-cfg.code_indent_scan_lines:] if len(lines) > cfg.code_indent_scan_lines else lines
    if not scan:
        return False
    indented = sum(1 for ln in scan if ln.startswith(cfg.code_indent_prefixes))
    ratio = indented / max(1, len(scan))
    return ratio >= cfg.code_indent_min_ratio


def split_into_paragraphs(text: str, rx: dict[str, re.Pattern[str]]) -> list[str]:
    t = norm_text(text, rx)
    if not t:
        return []
    return [p.strip() for p in t.split("\n\n") if p.strip()]


def looks_list_like(text: str) -> bool:
    lines = text.splitlines()
    if len(lines) < 2:
        return False
    markers = ("- ", "• ", "* ")
    return any(ln.lstrip().startswith(markers) for ln in lines)


def split_into_sentences(text: str, cfg: ChunkerConfig, rx: dict[str, re.Pattern[str]]) -> list[str]:
    t = norm_text(text, rx)
    if not t:
        return []
    if is_tableish_text(t, cfg, rx) or is_codeish_text(t, cfg, rx) or looks_list_like(t):
        return [t]
    parts = [p.strip() for p in rx["sent_end"].split(t) if p.strip()]
    return parts or [t]


def hard_cut(text: str, max_chars: int) -> list[str]:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return [t]
    out: list[str] = []
    cur = t
    while len(cur) > max_chars:
        cut = cur.rfind("\n\n", 0, max_chars)
        if cut < int(max_chars * 0.60):
            cut = cur.rfind("\n", 0, max_chars)
        if cut < int(max_chars * 0.60):
            cut = cur.rfind(" ", 0, max_chars)
        if cut < int(max_chars * 0.60):
            cut = max_chars
        out.append(cur[:cut].strip())
        cur = cur[cut:].strip()
    if cur:
        out.append(cur)
    return out


def split_best_effort(text: str, cfg: ChunkerConfig, rx: dict[str, re.Pattern[str]], max_chars: int) -> list[str]:
    t = norm_text(text, rx)
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]

    out: list[str] = []

    paras = split_into_paragraphs(t, rx)
    if len(paras) > 1:
        cur: list[str] = []
        cur_len = 0
        for p in paras:
            if cur_len and cur_len + 2 + len(p) > max_chars:
                out.append(join_nonempty(cur, rx))
                cur, cur_len = [], 0
            if len(p) > max_chars:
                if cur:
                    out.append(join_nonempty(cur, rx))
                    cur, cur_len = [], 0
                out.extend(split_best_effort(p, cfg, rx, max_chars))
                continue
            cur.append(p)
            cur_len += len(p) + 2
        if cur:
            out.append(join_nonempty(cur, rx))
        return [x for x in out if x]

    sents = split_into_sentences(t, cfg, rx)
    if len(sents) > 1:
        cur = ""
        for s in sents:
            if not cur:
                if len(s) <= max_chars:
                    cur = s
                else:
                    out.extend(hard_cut(s, max_chars))
                    cur = ""
                continue
            if len(cur) + 1 + len(s) <= max_chars:
                cur = f"{cur} {s}"
            else:
                out.append(cur.strip())
                if len(s) <= max_chars:
                    cur = s
                else:
                    out.extend(hard_cut(s, max_chars))
                    cur = ""
        if cur:
            out.append(cur.strip())
        return [x for x in out if x]

    return hard_cut(t, max_chars)


def build_overlap(text_body: str, cfg: ChunkerConfig, rx: dict[str, re.Pattern[str]]) -> str:
    t = norm_text(text_body, rx)
    if not t:
        return ""
    if is_tableish_text(t, cfg, rx) or is_codeish_text(t, cfg, rx):
        return ""

    paras = split_into_paragraphs(t, rx)
    base = paras[-1] if len(paras) >= 2 else t

    sents = split_into_sentences(base, cfg, rx)
    if not sents:
        return ""

    picked: list[str] = []
    for s in reversed(sents):
        s = s.strip()
        if not s:
            continue
        candidate = (" ".join(reversed(picked + [s]))).strip()
        if picked and len(candidate) > cfg.overlap_max_chars:
            break
        picked.append(s)
        if len(picked) >= cfg.overlap_max_sentences:
            break

    overlap = " ".join(reversed(picked)).strip()
    if overlap and len(overlap) > cfg.overlap_max_chars:
        overlap = overlap[: cfg.overlap_max_chars].rsplit(" ", 1)[0].strip()
    return overlap


def block_to_text(block: dict[str, Any], rx: dict[str, re.Pattern[str]]) -> str:
    t = (block.get("type") or "").strip()

    if t == "heading":
        return norm_text(block.get("text") or "", rx)
    if t in ("paragraph", "callout"):
        return norm_text(block.get("text") or "", rx)
    if t == "list":
        items = block.get("items") or []
        ordered = bool(block.get("ordered"))
        lines: list[str] = []
        if ordered:
            n = 1
            for it in items:
                itn = norm_text(it, rx)
                if itn:
                    lines.append(f"{n}. {itn}")
                    n += 1
        else:
            for it in items:
                itn = norm_text(it, rx)
                if itn:
                    lines.append(f"- {itn}")
        return "\n".join(lines).strip()
    if t in ("code", "table"):
        return (block.get("text") or "").rstrip()

    return norm_text(block.get("text") or "", rx)


def apply_heading_path(heading_stack: list[str], block: dict[str, Any], cfg: ChunkerConfig, rx: dict[str, re.Pattern[str]]) -> list[str]:
    level = int(block.get("level") or 1)
    text = norm_text(block.get("text") or "", rx)
    if not text:
        return heading_stack

    level = max(1, min(level, cfg.max_heading_level))
    idx = level - 1
    if len(heading_stack) <= idx:
        heading_stack.extend([""] * (idx + 1 - len(heading_stack)))
    heading_stack[idx] = text
    del heading_stack[idx + 1:]
    return [h for h in heading_stack if h]


def make_chunk_id(article_id: int, n: int) -> str:
    return f"{article_id:06d}-{n:04d}"


def compute_article_hash(article: dict[str, Any], content: str) -> str:
    payload = {
        "article_id": int(article.get("entry_id")),
        "source_url": article.get("source_url") or "",
        "title": article.get("title") or "",
        "revision": article.get("revision") or 0,
        "updated_at": article.get("updated_at") or "",
        "content": content or "",
    }
    return sha256_hex(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def compute_chunk_hash(article_meta: dict[str, Any], chunk: dict[str, Any]) -> str:
    payload = {
        "chunk_id": chunk["chunk_id"],
        "article_id": int(article_meta["article_id"]),
        "chunk_ord": int(chunk["chunk_ord"]),
        "section_path": chunk.get("section_path", []),
        "block_types": chunk.get("block_types", []),
        "text": chunk.get("text", ""),
        "images": chunk.get("images", []),
        "revision": article_meta.get("revision"),
        "updated_at_src": article_meta.get("updated_at_src"),
    }
    return sha256_hex(json.dumps(payload, ensure_ascii=False, sort_keys=True))


# ==============================
# Chunking core (returns plain dicts)
# ==============================

def chunk_article(article: dict[str, Any], cfg: ChunkerConfig, rx: dict[str, re.Pattern[str]]) -> tuple[list[dict[str, Any]], str]:
    """
    Returns (chunks, article_content_text)
    chunks: list of {chunk_id, chunk_ord, section_path, block_types, images, text, char_len}
    article_content_text: full normalized text of article (no images)
    """
    entry_id = int(article.get("entry_id"))
    blocks: list[dict[str, Any]] = article.get("blocks") or []

    chunks: list[dict[str, Any]] = []
    heading_stack: list[str] = []
    section_path: list[str] = []

    buf_parts: list[str] = []
    buf_types: list[str] = []
    buf_images: list[dict[str, Any]] = []
    buf_len = 0

    overlap_tail = ""
    chunk_index = 1

    pending_small_callout: Optional[tuple[str, list[dict[str, Any]]]] = None

    # Build article-level content too
    article_text_parts: list[str] = []

    def flush_buf_if_any() -> None:
        nonlocal chunk_index, overlap_tail, buf_parts, buf_types, buf_images, buf_len

        text_body = join_nonempty(buf_parts, rx)
        if not text_body and not buf_images:
            buf_parts, buf_types, buf_images, buf_len = [], [], [], 0
            return

        text = text_body
        if overlap_tail:
            text = join_nonempty([f"{cfg.overlap_prefix}{overlap_tail}", text_body], rx) if text_body else f"{cfg.overlap_prefix}{overlap_tail}"

        chunks.append({
            "chunk_id": make_chunk_id(entry_id, chunk_index),
            "chunk_ord": chunk_index,
            "section_path": section_path.copy(),
            "block_types": buf_types.copy(),
            "images": dedupe_images(buf_images),
            "text": text,
            "char_len": len(text),
        })
        chunk_index += 1

        overlap_tail = build_overlap(text_body, cfg, rx)

        buf_parts, buf_types, buf_images, buf_len = [], [], [], 0

    def buf_add(text: str, btype: str, imgs: list[dict[str, Any]]) -> None:
        nonlocal buf_len
        if not text and not imgs:
            return
        if text:
            buf_parts.append(text)
            buf_types.append(btype)
            buf_len += len(text)
        if imgs:
            buf_images.extend(imgs)

    def emit_as_own_chunks(btype: str, btxt: str, bimgs: list[dict[str, Any]]) -> None:
        nonlocal chunk_index, overlap_tail
        parts = split_best_effort(btxt, cfg, rx, cfg.max_chars) if btxt else [""]
        imgs = dedupe_images(bimgs)
        for part in parts:
            part = part.strip()
            if not part and not imgs:
                continue
            text = join_nonempty([f"{cfg.overlap_prefix}{overlap_tail}", part], rx) if overlap_tail else part
            chunks.append({
                "chunk_id": make_chunk_id(entry_id, chunk_index),
                "chunk_ord": chunk_index,
                "section_path": section_path.copy(),
                "block_types": [btype],
                "images": imgs,
                "text": text,
                "char_len": len(text),
            })
            chunk_index += 1
            overlap_tail = build_overlap(part, cfg, rx)

    def maybe_attach_pending_callout_to(text: str, imgs: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        nonlocal pending_small_callout
        if not pending_small_callout:
            return text, imgs
        c_text, c_imgs = pending_small_callout
        pending_small_callout = None
        merged_text = join_nonempty([c_text, text], rx) if c_text else text
        merged_imgs = dedupe_images((c_imgs or []) + (imgs or []))
        return merged_text, merged_imgs

    def emit_pending_callout_if_any() -> None:
        nonlocal pending_small_callout, chunk_index, overlap_tail
        if not pending_small_callout:
            return
        c_text, c_imgs = pending_small_callout
        pending_small_callout = None
        flush_buf_if_any()
        chunks.append({
            "chunk_id": make_chunk_id(entry_id, chunk_index),
            "chunk_ord": chunk_index,
            "section_path": section_path.copy(),
            "block_types": ["callout"],
            "images": dedupe_images(c_imgs),
            "text": norm_text(c_text, rx),
            "char_len": len(norm_text(c_text, rx)),
        })
        chunk_index += 1
        overlap_tail = ""

    def can_inline_atomic(btype: str, btxt: str) -> bool:
        if not cfg.allow_inline_atomic_in_buffer:
            return False
        if btype not in ("code", "table"):
            return False
        return len(btxt or "") <= cfg.inline_atomic_max_chars

    for b in blocks:
        btype = (b.get("type") or "").strip()

        if btype == "heading":
            emit_pending_callout_if_any()
            flush_buf_if_any()
            heading_stack = apply_heading_path(heading_stack, b, cfg, rx)
            section_path = heading_stack.copy()
            continue

        btxt = block_to_text(b, rx)
        bimgs = block_images(b)

        if not btxt and not bimgs:
            continue

        # build article-level text (no overlap prefix, no image anchors)
        if btxt:
            article_text_parts.append(btxt)

        if cfg.merge_small_callouts and btype == "callout":
            c_text = norm_text(btxt, rx)
            if c_text and len(c_text) <= cfg.callout_merge_max_chars:
                pending_small_callout = (c_text, bimgs)
                continue
            emit_pending_callout_if_any()
            flush_buf_if_any()
            emit_as_own_chunks("callout", btxt, bimgs)
            overlap_tail = ""
            continue

        if pending_small_callout:
            btxt, bimgs = maybe_attach_pending_callout_to(btxt, bimgs)

        if btxt and len(btxt) > cfg.max_chars:
            flush_buf_if_any()
            emit_as_own_chunks(btype, btxt, bimgs)
            continue

        if btype in ("code", "table"):
            if can_inline_atomic(btype, btxt) and (buf_len + 2 + len(btxt) <= cfg.max_chars):
                buf_add(btxt, btype, bimgs)
                if buf_len >= cfg.target_chars:
                    flush_buf_if_any()
            else:
                flush_buf_if_any()
                emit_as_own_chunks(btype, btxt, bimgs)
                overlap_tail = ""
            continue

        if buf_len and btxt and (buf_len + 2 + len(btxt)) > cfg.max_chars:
            flush_buf_if_any()

        buf_add(btxt, btype, bimgs)

        if buf_len >= cfg.target_chars:
            flush_buf_if_any()

    emit_pending_callout_if_any()
    flush_buf_if_any()

    # post-merge tiny neighboring chunks (same section)
    if cfg.merge_tiny_chunks:
        merged: list[dict[str, Any]] = []
        i = 0
        while i < len(chunks):
            cur = chunks[i]
            if (
                i + 1 < len(chunks)
                and len(cur["text"]) < cfg.min_chars
                and len(cur["text"]) + 2 + len(chunks[i + 1]["text"]) <= cfg.max_chars
                and cur["section_path"] == chunks[i + 1]["section_path"]
            ):
                nxt = chunks[i + 1]
                merged_text = join_nonempty([cur["text"], nxt["text"]], rx)
                merged_imgs = dedupe_images((cur.get("images") or []) + (nxt.get("images") or []))
                merged.append({
                    "chunk_id": cur["chunk_id"],  # temporary
                    "chunk_ord": cur["chunk_ord"],  # temporary
                    "section_path": cur["section_path"],
                    "block_types": (cur.get("block_types") or []) + (nxt.get("block_types") or []),
                    "images": merged_imgs,
                    "text": merged_text,
                    "char_len": len(merged_text),
                })
                i += 2
            else:
                merged.append(cur)
                i += 1

        # renumber chunk_ord & chunk_id again
        out: list[dict[str, Any]] = []
        for n, ch in enumerate(merged, start=1):
            out.append({
                **ch,
                "chunk_id": make_chunk_id(entry_id, n),
                "chunk_ord": n,
            })
        chunks = out

    article_content = join_nonempty(article_text_parts, rx, sep="\n\n")
    return chunks, article_content


# ==============================
# DB load
# ==============================

SQL_UPSERT_ARTICLE = f"""
INSERT INTO {PG_SCHEMA}.articles(article_id, source_url, title, content, revision, article_hash, updated_at_src)
VALUES (%(article_id)s, %(source_url)s, %(title)s, %(content)s, %(revision)s, %(article_hash)s, %(updated_at_src)s)
ON CONFLICT (article_id) DO UPDATE SET
  source_url     = EXCLUDED.source_url,
  title          = EXCLUDED.title,
  content        = EXCLUDED.content,
  revision       = EXCLUDED.revision,
  article_hash   = EXCLUDED.article_hash,
  updated_at_src = EXCLUDED.updated_at_src,
  updated_at     = now();
"""

SQL_UPSERT_CHUNK = f"""
INSERT INTO {PG_SCHEMA}.chunks(chunk_id, article_id, chunk_ord, section_path, block_types, text, char_len, chunk_hash)
VALUES (
  %(chunk_id)s, %(article_id)s, %(chunk_ord)s,
  %(section_path)s::jsonb, %(block_types)s::jsonb,
  %(text)s, %(char_len)s, %(chunk_hash)s
)
ON CONFLICT (chunk_id) DO UPDATE SET
  article_id   = EXCLUDED.article_id,
  chunk_ord    = EXCLUDED.chunk_ord,
  section_path = EXCLUDED.section_path,
  block_types  = EXCLUDED.block_types,
  text         = EXCLUDED.text,
  char_len     = EXCLUDED.char_len,
  chunk_hash   = EXCLUDED.chunk_hash,
  updated_at   = now();
"""

SQL_UPSERT_IMAGE = f"""
INSERT INTO {PG_SCHEMA}.images(local_path, alt, caption)
VALUES (%(local_path)s, %(alt)s, %(caption)s)
ON CONFLICT (local_path) DO UPDATE SET
  alt = EXCLUDED.alt,
  caption = EXCLUDED.caption
RETURNING image_id;
"""

SQL_DELETE_CHUNK_IMAGES = f"""
DELETE FROM {PG_SCHEMA}.chunk_images WHERE chunk_id = %(chunk_id)s;
"""

SQL_INSERT_CHUNK_IMAGE = f"""
INSERT INTO {PG_SCHEMA}.chunk_images(chunk_id, image_id, ord)
VALUES (%(chunk_id)s, %(image_id)s, %(ord)s)
ON CONFLICT (chunk_id, ord) DO UPDATE SET
  image_id = EXCLUDED.image_id;
"""


def iter_entry_dirs(in_root: Path) -> list[Path]:
    if not in_root.exists():
        raise SystemExit(f"IN_ROOT not found: {in_root}")
    return sorted([p for p in in_root.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))


def load_article_json(entry_dir: Path) -> Optional[dict[str, Any]]:
    p = entry_dir / "article.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def run() -> None:
    import psycopg2

    cfg = ChunkerConfig()
    rx = build_regexes(cfg)

    total_entries = 0
    total_chunks = 0

    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        cur.execute("SET statement_timeout = 0;")

        for entry_dir in iter_entry_dirs(IN_ROOT):
            article = load_article_json(entry_dir)
            if not article:
                continue

            article_id = int(article.get("entry_id"))
            source_url = article.get("source_url")
            title = article.get("title")
            revision = article.get("revision")
            updated_at_src = article.get("updated_at")

            chunks, article_content = chunk_article(article, cfg, rx)
            if not chunks:
                continue

            article_hash = compute_article_hash(article, article_content)

            # upsert article
            cur.execute(SQL_UPSERT_ARTICLE, {
                "article_id": article_id,
                "source_url": source_url,
                "title": title,
                "content": article_content,
                "revision": revision,
                "article_hash": article_hash,
                "updated_at_src": updated_at_src,
            })

            # upsert chunks + images
            for ch in chunks:
                c_hash = compute_chunk_hash({
                    "article_id": article_id,
                    "revision": revision,
                    "updated_at_src": updated_at_src,
                }, ch)

                cur.execute(SQL_UPSERT_CHUNK, {
                    "chunk_id": ch["chunk_id"],
                    "article_id": article_id,
                    "chunk_ord": ch["chunk_ord"],
                    "section_path": json.dumps(ch.get("section_path", []), ensure_ascii=False),
                    "block_types": json.dumps(ch.get("block_types", []), ensure_ascii=False),
                    "text": ch["text"],
                    "char_len": ch["char_len"],
                    "chunk_hash": c_hash,
                })

                images = ch.get("images") or []
                if images:
                    cur.execute(SQL_DELETE_CHUNK_IMAGES, {"chunk_id": ch["chunk_id"]})
                    for idx, im in enumerate(images, start=1):
                        local_path = (im.get("local_path") or "").strip()
                        if not local_path:
                            continue

                        cur.execute(SQL_UPSERT_IMAGE, {
                            "local_path": local_path,
                            "alt": im.get("alt"),
                            "caption": im.get("caption"),
                        })
                        image_id = cur.fetchone()[0]
                        cur.execute(SQL_INSERT_CHUNK_IMAGE, {
                            "chunk_id": ch["chunk_id"],
                            "image_id": image_id,
                            "ord": idx,
                        })

            total_entries += 1
            total_chunks += len(chunks)

            if total_entries % 50 == 0:
                conn.commit()
                print(f"[OK] entries={total_entries}, chunks={total_chunks}")

        conn.commit()
        print(f"[DONE] entries={total_entries}, chunks={total_chunks}")

    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    run()
