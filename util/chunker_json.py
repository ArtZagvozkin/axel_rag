#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chunker for sanitized KBPublisher articles (clean_data/entries/<id>/article.json).

Design goals (no hardcoded domain heuristics):
- Chunking is driven by *document structure* (block types, paragraphs, sentences) and configurable limits.
- Any “heuristics” are expressed as config thresholds/regexes and can be overridden via JSON config file or CLI flags.
- Do NOT put image paths into chunk text. All image refs go to chunk metadata ("images").
- Avoid “обрубыши”: overlap is built only from complete units (paragraphs / sentences).

Outputs:
- clean_data/chunks/chunks.jsonl
- optional clean_data/chunks/chunks_preview.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional, Iterable


# ==============================
# CONFIG (all tunables are here)
# ==============================

@dataclass
class ChunkerConfig:
    # IO
    in_root: str = "clean_data/entries"
    out_dir: str = "clean_data/chunks"
    out_jsonl: str = "chunks.jsonl"
    out_preview_md: Optional[str] = "chunks_preview.md"

    # Filtering
    only_entry_ids: list[int] = None  # type: ignore

    # Sizes
    target_chars: int = 2300
    max_chars: int = 3000
    min_chars: int = 750

    # Overlap (only full sentences/paragraphs)
    overlap_max_chars: int = 200
    overlap_max_sentences: int = 1
    overlap_prefix: str = "(контекст) "

    # Merging small blocks into neighbors (structure-based, not content-based)
    merge_small_callouts: bool = True
    callout_merge_max_chars: int = 260

    # Allow keeping short code/table blocks together with surrounding text in the buffer
    # (purely size-based; no “detect instruction text”)
    allow_inline_atomic_in_buffer: bool = True
    inline_atomic_max_chars: int = 420  # if atomic block text <= this and buffer has room, keep together

    # Post-merge tiny neighboring chunks (same section)
    merge_tiny_chunks: bool = True

    # Text normalization
    ws_re: str = r"[ \t\r\f\v]+"
    nl3_re: str = r"\n{3,}"

    # Sentence boundary regex (conservative, configurable)
    # Note: keep it simple; avoid domain-specific abbreviation lists.
    sent_end_re: str = r"(?<=[.!?])\s+"

    # Table detection (structural)
    md_table_row_re: str = r"^\s*\|.*\|\s*$"
    md_table_min_rows: int = 3
    md_table_scan_lines: int = 10

    # Code-ish detection (structural)
    # A text is “code-ish” if enough of its recent lines are indented OR it contains fenced code.
    code_fence_re: str = r"(^```|^~~~)"
    code_indent_min_ratio: float = 0.35
    code_indent_scan_lines: int = 30
    code_indent_prefixes: tuple[str, ...] = ("    ", "\t")

    # Heading stack
    max_heading_level: int = 6


def load_config(path: Optional[str]) -> ChunkerConfig:
    cfg = ChunkerConfig(only_entry_ids=[])
    if not path:
        return cfg
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    # allow partial overrides
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    # normalize None -> []
    if cfg.only_entry_ids is None:
        cfg.only_entry_ids = []
    return cfg


# ==============================
# Helpers
# ==============================

def approx_tokens(chars: int) -> int:
    # stable approximation
    return max(1, math.ceil(chars / 4))


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
    """Deduplicate by local_path (keep first occurrence)."""
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
    scan = lines[-cfg.code_indent_scan_lines :] if len(lines) > cfg.code_indent_scan_lines else lines
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
    # purely structural: multiple lines and some common list markers at line starts
    lines = text.splitlines()
    if len(lines) < 2:
        return False
    markers = ("- ", "• ", "* ")
    return any(ln.lstrip().startswith(markers) for ln in lines)


def split_into_sentences(text: str, cfg: ChunkerConfig, rx: dict[str, re.Pattern[str]]) -> list[str]:
    """
    Conservative sentence splitting. No domain abbreviations.
    If text is table/code/list-ish, return as one unit.
    """
    t = norm_text(text, rx)
    if not t:
        return []
    if is_tableish_text(t, cfg, rx) or is_codeish_text(t, cfg, rx) or looks_list_like(t):
        return [t]
    parts = [p.strip() for p in rx["sent_end"].split(t) if p.strip()]
    return parts or [t]


def hard_cut(text: str, max_chars: int) -> list[str]:
    """
    Last resort cutting; tries to cut at nearest whitespace/newline.
    Still avoids splitting words when possible.
    """
    t = (text or "").strip()
    if len(t) <= max_chars:
        return [t]
    out: list[str] = []
    cur = t
    while len(cur) > max_chars:
        # prefer paragraph break, then line break, then space
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
    """
    Split without cutting sentences if possible.
    Order:
    1) pack paragraphs
    2) if single huge paragraph -> pack sentences
    3) hard cut as last resort
    """
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
    """
    Build overlap from end of text_body using complete units:
    - Prefer last paragraph if multi-paragraph.
    - Else last 1..N sentences.
    - Never use overlap for code/table-ish bodies.
    """
    t = norm_text(text_body, rx)
    if not t:
        return ""

    if is_tableish_text(t, cfg, rx) or is_codeish_text(t, cfg, rx):
        return ""

    paras = split_into_paragraphs(t, rx)
    if len(paras) >= 2:
        tail = paras[-1]
        if len(tail) <= cfg.overlap_max_chars:
            return tail
        # fall through to sentence-based overlap on tail
        base = tail
    else:
        base = t

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
    """
    Convert block to text WITHOUT image anchors.
    Images are handled separately and stored in chunk metadata.
    """
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
    del heading_stack[idx + 1 :]
    return [h for h in heading_stack if h]


def make_chunk_id(entry_id: int, n: int) -> str:
    return f"{entry_id:06d}-{n:04d}"


# ==============================
# Chunk model
# ==============================

@dataclass
class Chunk:
    chunk_id: str
    entry_id: int
    source_url: Optional[str]
    title: Optional[str]
    updated_at: Optional[str]
    revision: Optional[int]
    section_path: list[str]
    block_types: list[str]
    images: list[dict[str, Any]]
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "entry_id": self.entry_id,
            "source_url": self.source_url,
            "title": self.title,
            "updated_at": self.updated_at,
            "revision": self.revision,
            "section_path": self.section_path,
            "block_types": self.block_types,
            "images": self.images,
            "text": self.text,
            "char_len": len(self.text),
            "approx_tokens": approx_tokens(len(self.text)),
        }


# ==============================
# Core chunking
# ==============================

def chunk_article(article: dict[str, Any], cfg: ChunkerConfig, rx: dict[str, re.Pattern[str]]) -> list[Chunk]:
    entry_meta = {
        "entry_id": article.get("entry_id"),
        "source_url": article.get("source_url"),
        "title": article.get("title"),
        "updated_at": article.get("updated_at"),
        "revision": article.get("revision"),
    }
    entry_id = int(entry_meta["entry_id"])
    blocks: list[dict[str, Any]] = article.get("blocks") or []

    chunks: list[Chunk] = []
    heading_stack: list[str] = []
    section_path: list[str] = []

    # buffer collects *logical blocks* (whole block texts). splitting is only inside a block if it exceeds max_chars.
    buf_parts: list[str] = []
    buf_types: list[str] = []
    buf_images: list[dict[str, Any]] = []
    buf_len = 0

    overlap_tail = ""
    chunk_index = 1

    pending_small_callout: Optional[tuple[str, list[dict[str, Any]]]] = None

    def flush_buf_if_any() -> None:
        nonlocal chunk_index, overlap_tail, buf_parts, buf_types, buf_images, buf_len

        text_body = join_nonempty(buf_parts, rx)
        if not text_body and not buf_images:
            buf_parts, buf_types, buf_images, buf_len = [], [], [], 0
            return

        text = text_body
        if overlap_tail:
            text = join_nonempty([f"{cfg.overlap_prefix}{overlap_tail}", text_body], rx) if text_body else f"{cfg.overlap_prefix}{overlap_tail}"

        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(entry_id, chunk_index),
                entry_id=entry_id,
                source_url=entry_meta.get("source_url"),
                title=entry_meta.get("title"),
                updated_at=entry_meta.get("updated_at"),
                revision=entry_meta.get("revision"),
                section_path=section_path.copy(),
                block_types=buf_types.copy(),
                images=dedupe_images(buf_images),
                text=text,
            )
        )
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
        """Split a big block and emit parts as separate chunks with overlap."""
        nonlocal chunk_index, overlap_tail
        parts = split_best_effort(btxt, cfg, rx, cfg.max_chars) if btxt else [""]
        imgs = dedupe_images(bimgs)
        for part in parts:
            part = part.strip()
            if not part and not imgs:
                continue
            text = join_nonempty([f"{cfg.overlap_prefix}{overlap_tail}", part], rx) if overlap_tail else part
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(entry_id, chunk_index),
                    entry_id=entry_id,
                    source_url=entry_meta.get("source_url"),
                    title=entry_meta.get("title"),
                    updated_at=entry_meta.get("updated_at"),
                    revision=entry_meta.get("revision"),
                    section_path=section_path.copy(),
                    block_types=[btype],
                    images=imgs,
                    text=text,
                )
            )
            chunk_index += 1
            overlap_tail = build_overlap(part, cfg, rx)

    def maybe_attach_pending_callout_to(text: str, imgs: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """If there is a pending small callout, attach it structurally (prepend)."""
        nonlocal pending_small_callout
        if not pending_small_callout:
            return text, imgs
        c_text, c_imgs = pending_small_callout
        pending_small_callout = None
        merged_text = join_nonempty([c_text, text], rx) if c_text else text
        merged_imgs = dedupe_images((c_imgs or []) + (imgs or []))
        return merged_text, merged_imgs

    def emit_pending_callout_if_any() -> None:
        """If pending callout exists, emit it as its own chunk (no special heuristics)."""
        nonlocal pending_small_callout, chunk_index, overlap_tail
        if not pending_small_callout:
            return
        c_text, c_imgs = pending_small_callout
        pending_small_callout = None
        flush_buf_if_any()
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(entry_id, chunk_index),
                entry_id=entry_id,
                source_url=entry_meta.get("source_url"),
                title=entry_meta.get("title"),
                updated_at=entry_meta.get("updated_at"),
                revision=entry_meta.get("revision"),
                section_path=section_path.copy(),
                block_types=["callout"],
                images=dedupe_images(c_imgs),
                text=norm_text(c_text, rx),
            )
        )
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

        # Small callouts: structural merge into next non-heading block if enabled.
        if cfg.merge_small_callouts and btype == "callout":
            c_text = norm_text(btxt, rx)
            if c_text and len(c_text) <= cfg.callout_merge_max_chars:
                pending_small_callout = (c_text, bimgs)
                continue
            # large callout: emit separately
            emit_pending_callout_if_any()
            flush_buf_if_any()
            emit_as_own_chunks("callout", btxt, bimgs)
            overlap_tail = ""
            continue

        # Attach pending small callout to the next block (any type except heading)
        if pending_small_callout:
            btxt, bimgs = maybe_attach_pending_callout_to(btxt, bimgs)

        # If block itself is too large, emit it as separate split chunks
        if btxt and len(btxt) > cfg.max_chars:
            flush_buf_if_any()
            emit_as_own_chunks(btype, btxt, bimgs)
            continue

        # Atomic blocks:
        # - by default they can be kept inline if short and buffer has space (structure-based).
        # - otherwise flush and emit as own chunk(s).
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

        # Regular text blocks: pack whole blocks into buffer; flush before overflow.
        if buf_len and btxt and (buf_len + 2 + len(btxt)) > cfg.max_chars:
            flush_buf_if_any()

        buf_add(btxt, btype, bimgs)

        if buf_len >= cfg.target_chars:
            flush_buf_if_any()

    # end
    emit_pending_callout_if_any()
    flush_buf_if_any()

    # Post-merge tiny neighboring chunks (same section), unless they include large atomics.
    # This is also structure-based and controlled by config.
    if not cfg.merge_tiny_chunks:
        # renumber and return
        return _renumber(entry_id, chunks)

    merged: list[Chunk] = []
    i = 0
    while i < len(chunks):
        cur = chunks[i]
        if (
            i + 1 < len(chunks)
            and len(cur.text) < cfg.min_chars
            and len(cur.text) + 2 + len(chunks[i + 1].text) <= cfg.max_chars
            and cur.entry_id == chunks[i + 1].entry_id
            and cur.section_path == chunks[i + 1].section_path
        ):
            nxt = chunks[i + 1]
            merged_text = join_nonempty([cur.text, nxt.text], rx)
            merged_imgs = dedupe_images((cur.images or []) + (nxt.images or []))
            merged.append(
                Chunk(
                    chunk_id=cur.chunk_id,  # temporary
                    entry_id=cur.entry_id,
                    source_url=cur.source_url,
                    title=cur.title,
                    updated_at=cur.updated_at,
                    revision=cur.revision,
                    section_path=cur.section_path,
                    block_types=cur.block_types + nxt.block_types,
                    images=merged_imgs,
                    text=merged_text,
                )
            )
            i += 2
        else:
            merged.append(cur)
            i += 1

    return _renumber(entry_id, merged)


def _renumber(entry_id: int, chunks: list[Chunk]) -> list[Chunk]:
    out: list[Chunk] = []
    for n, ch in enumerate(chunks, start=1):
        out.append(
            Chunk(
                chunk_id=make_chunk_id(entry_id, n),
                entry_id=ch.entry_id,
                source_url=ch.source_url,
                title=ch.title,
                updated_at=ch.updated_at,
                revision=ch.revision,
                section_path=ch.section_path,
                block_types=ch.block_types,
                images=ch.images,
                text=ch.text,
            )
        )
    return out


# ==============================
# IO / main
# ==============================

def iter_entry_dirs(in_root: Path, only_entry_ids: list[int]) -> list[Path]:
    if not in_root.exists():
        raise SystemExit(f"IN_ROOT not found: {in_root}")
    dirs = sorted([p for p in in_root.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
    if only_entry_ids:
        allowed = set(int(x) for x in only_entry_ids)
        dirs = [p for p in dirs if int(p.name) in allowed]
    return dirs


def dump_default_config(path: str) -> None:
    cfg = ChunkerConfig(only_entry_ids=[])
    Path(path).write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote default config to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Chunker for KBPublisher sanitized articles")
    ap.add_argument("--config", type=str, default=None, help="Path to JSON config overrides")
    ap.add_argument("--dump-config", type=str, default=None, help="Write default config JSON to this path and exit")

    # Quick overrides (optional)
    ap.add_argument("--max-chars", type=int, default=None)
    ap.add_argument("--target-chars", type=int, default=None)
    ap.add_argument("--min-chars", type=int, default=None)
    ap.add_argument("--only", type=str, default=None, help="Comma-separated entry ids, e.g. 12,13,77")

    args = ap.parse_args()

    if args.dump_config:
        dump_default_config(args.dump_config)
        return

    cfg = load_config(args.config)

    if args.max_chars is not None:
        cfg.max_chars = int(args.max_chars)
    if args.target_chars is not None:
        cfg.target_chars = int(args.target_chars)
    if args.min_chars is not None:
        cfg.min_chars = int(args.min_chars)
    if args.only:
        cfg.only_entry_ids = [int(x.strip()) for x in args.only.split(",") if x.strip().isdigit()]

    rx = build_regexes(cfg)

    in_root = Path(cfg.in_root)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl_path = out_dir / cfg.out_jsonl
    out_md_path = (out_dir / cfg.out_preview_md) if cfg.out_preview_md else None

    total_entries = 0
    total_chunks = 0

    with out_jsonl_path.open("w", encoding="utf-8") as f_jsonl, (
        out_md_path.open("w", encoding="utf-8") if out_md_path else open(Path("/dev/null"), "w", encoding="utf-8")
    ) as f_md:

        if out_md_path:
            f_md.write("# Chunks preview\n\n")

        for entry_dir in iter_entry_dirs(in_root, cfg.only_entry_ids or []):
            article_path = entry_dir / "article.json"
            if not article_path.exists():
                continue

            article = json.loads(article_path.read_text(encoding="utf-8"))
            chunks = chunk_article(article, cfg, rx)
            if not chunks:
                continue

            total_entries += 1
            total_chunks += len(chunks)

            for ch in chunks:
                f_jsonl.write(json.dumps(ch.to_dict(), ensure_ascii=False) + "\n")

            if out_md_path:
                f_md.write(f"\n---\n\n## Entry {article.get('entry_id')} — {article.get('title','')}\n\n")
                f_md.write(f"Источник: {article.get('source_url','')}\n\n")
                for ch in chunks:
                    sp = " / ".join(ch.section_path) if ch.section_path else "(no headings)"
                    f_md.write(f"### {ch.chunk_id}  ({len(ch.text)} chars)\n\n")
                    f_md.write(f"- Section: {sp}\n")
                    f_md.write(f"- Types: {', '.join(ch.block_types)}\n")
                    f_md.write(f"- Images: {len(ch.images)}\n\n")
                    f_md.write(ch.text.replace("\n", "\n\n") + "\n\n")

            if total_entries % 50 == 0:
                print(f"[OK] processed {total_entries} entries, chunks={total_chunks}")

    print(f"[DONE] entries={total_entries}, chunks={total_chunks}")
    print(f"Output: {out_dir}")
    print(f"- {out_jsonl_path}")
    if out_md_path:
        print(f"- {out_md_path}")


if __name__ == "__main__":
    main()
