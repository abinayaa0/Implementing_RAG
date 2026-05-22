"""
chunker.py — Works Manual SE 2025 — Fixed section detection
"""

import re
import json
import pdfplumber
from pathlib import Path

PDF_PATH    = "Works_Manual_SE_2025.pdf"
OUTPUT_PATH = "/home/claude/chunks2.json"

CHUNK_SIZE          = 800
CHUNK_OVERLAP       = 100
AVG_CHARS_PER_TOKEN = 4


# ── Extract ────────────────────────────────────────────────────────────────────
def extract_text_by_page(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            pages.append({"page": i + 1, "text": page.extract_text() or ""})
    return pages

def build_full_text(pages):
    return "\n".join(f"\n<<<PAGE:{p['page']}>>>\n{p['text']}" for p in pages)


# ── Front matter ───────────────────────────────────────────────────────────────
def extract_front_matter_chunk(text):
    start = text.find("FOREWORD")
    end = len(text)
    for m in ["Table of Contents", "Chapter 1:"]:
        idx = text.find(m)
        if idx != -1 and idx > start:
            end = min(end, idx)
    content = clean_text(text[start:end])
    return {
        "chunk_id": "front_matter_0000",
        "chapter_num": "FM", "chapter_title": "Foreword and Preface",
        "section": "", "section_title": "Front Matter",
        "text": content[:3200],
        "char_count": len(content),
        "token_count": len(content) // AVG_CHARS_PER_TOKEN,
        "page_hint": 1,
    }

def strip_front_matter(text):
    first = text.find("Chapter 1:")
    second = text.find("Chapter 1:", first + 1)
    return text[second if second != -1 else first:]


# ── Regex (FIXED) ──────────────────────────────────────────────────────────────
# Chapter header:  "Chapter 3: Bidding Design for Works"
CHAPTER_RE = re.compile(r"^Chapter\s+(\d+)\s*[:\-\u2013]\s*(.+)$", re.IGNORECASE)

# Section header: "1.11.2 Procurement Preference to Make in India"
# FIX: was requiring 2+ spaces — PDF uses single space. Also relaxed title start.
SECTION_RE = re.compile(r"^(\d+\.\d+(?:\.\d+)?)\s+(.{4,90})$")

# Annexure header
ANNEXURE_RE = re.compile(r"^(Annexure\s+[\dA-Z]+[:\-\u2013]?\s*.{0,60})$", re.IGNORECASE)

# Lines to ignore as false section matches (e.g. "8.7 of this manual)")
FALSE_SECTION_RE = re.compile(r"^\d+\.\d+.*?(of this manual|above|below|refer|see para)", re.IGNORECASE)


def clean_text(text):
    text = re.sub(r"<<<PAGE:\d+>>>", "", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\.{4,}\s*\d+", "", text)
    text = re.sub(r"[^\x00-\x7F]{3,}", "", text)
    return text.strip()


# ── Structural segmentation ────────────────────────────────────────────────────
def split_into_segments(body_text):
    segments = []
    cur = dict(chapter_num="", chapter_title="",
               section="", section_title="", page_hint=1)
    buffer = []

    def flush():
        content = clean_text("\n".join(buffer))
        if len(content) > 80:
            seg = dict(cur)
            seg["text"] = content
            segments.append(seg)
        buffer.clear()

    for line in body_text.split("\n"):
        pm = re.search(r"<<<PAGE:(\d+)>>>", line)
        if pm:
            cur["page_hint"] = int(pm.group(1))
            continue

        stripped = line.strip()

        # Chapter
        cm = CHAPTER_RE.match(stripped)
        if cm:
            flush()
            cur.update(chapter_num=cm.group(1),
                       chapter_title=cm.group(2).strip(),
                       section="",
                       section_title=f"Chapter {cm.group(1)} Overview")
            buffer.append(line)
            continue

        # Annexure
        am = ANNEXURE_RE.match(stripped)
        if am and len(stripped) > 8:
            flush()
            title = am.group(1).strip()
            cur.update(chapter_num="Annexure", chapter_title=title,
                       section="", section_title=title)
            buffer.append(line)
            continue

        # Section — fixed: single space, relaxed title, false-match guard
        sm = SECTION_RE.match(stripped)
        if sm and not FALSE_SECTION_RE.match(stripped):
            sec_num = sm.group(1)
            expected = cur["chapter_num"] + "."
            if cur["chapter_num"] and sec_num.startswith(expected):
                flush()
                cur.update(section=sec_num,
                           section_title=sm.group(2).strip())
                buffer.append(line)
                continue

        buffer.append(line)

    flush()
    return segments


# ── Sub-chunking ───────────────────────────────────────────────────────────────
def token_count(text):
    return len(text) // AVG_CHARS_PER_TOKEN

def sub_chunk(segment):
    text = segment["text"]
    target_chars  = CHUNK_SIZE    * AVG_CHARS_PER_TOKEN
    overlap_chars = CHUNK_OVERLAP * AVG_CHARS_PER_TOKEN

    if len(text) <= target_chars:
        return [segment]

    for sep in ["\n\n", "\n", ". ", " "]:
        parts = text.split(sep)
        if len(parts) > 1:
            break

    raw_chunks, current = [], ""
    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) <= target_chars:
            current = candidate
        else:
            if current:
                raw_chunks.append(current.strip())
            if len(part) > target_chars * 1.5:
                while len(part) > target_chars:
                    raw_chunks.append(part[:target_chars].strip())
                    part = part[target_chars - overlap_chars:]
            current = part
    if current.strip():
        raw_chunks.append(current.strip())

    result = []
    for j, chunk_text in enumerate(raw_chunks):
        if j > 0:
            chunk_text = raw_chunks[j-1][-overlap_chars:] + "\n" + chunk_text
        sub = dict(segment)
        sub["text"] = clean_text(chunk_text)
        result.append(sub)
    return result


# ── IDs ────────────────────────────────────────────────────────────────────────
def assign_ids(chunks):
    for i, c in enumerate(chunks):
        sec_slug = c["section"].replace(".", "_") or "00"
        c["chunk_id"]    = f"ch{c['chapter_num']}_sec{sec_slug}_{i:04d}"
        c["char_count"]  = len(c["text"])
        c["token_count"] = token_count(c["text"])
    return chunks


# ── Main ───────────────────────────────────────────────────────────────────────
def run_pipeline():
    print("📄 Extracting text...")
    pages     = extract_text_by_page(PDF_PATH)
    full_text = build_full_text(pages)
    print(f"   {len(pages)} pages | {len(full_text):,} chars")

    fm   = extract_front_matter_chunk(full_text)
    body = strip_front_matter(full_text)

    print("🔍 Structural segmentation...")
    segments = split_into_segments(body)

    # Count section hits for diagnostics
    with_sec = [s for s in segments if s["section"]]
    print(f"   {len(segments)} segments | {len(with_sec)} have section numbers")

    print("📦 Sub-chunking...")
    all_chunks = [fm]
    for seg in segments:
        all_chunks.extend(sub_chunk(seg))

    all_chunks = assign_ids(all_chunks)

    toks     = [c["token_count"] for c in all_chunks]
    chapters = sorted(set(c["chapter_num"] for c in all_chunks))
    with_sec_chunks = [c for c in all_chunks if c["section"]]

    print(f"\n{'='*55}")
    print(f"  ✅  PIPELINE COMPLETE")
    print(f"{'='*55}")
    print(f"  Total chunks          : {len(all_chunks)}")
    print(f"  Chunks with section # : {len(with_sec_chunks)}")
    print(f"  Avg / Min / Max tokens: {sum(toks)/len(toks):.0f} / {min(toks)} / {max(toks)}")
    print(f"  Chapters found        : {chapters}")
    print(f"{'='*55}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"\n💾 {OUTPUT_PATH}  ({Path(OUTPUT_PATH).stat().st_size/1024:.1f} KB)")

    print("\n─── Sample section-level chunks ───")
    for c in with_sec_chunks[:5]:
        print(f"\n  {c['chunk_id']}")
        print(f"  Chapter : {c['chapter_num']} — {c['chapter_title']}")
        print(f"  Section : {c['section']} — {c['section_title']}")
        print(f"  Tokens  : {c['token_count']}  |  Page: {c['page_hint']}")
        print(f"  Preview : {c['text'][:180].strip()!r}")

    return all_chunks

if __name__ == "__main__":
    run_pipeline()