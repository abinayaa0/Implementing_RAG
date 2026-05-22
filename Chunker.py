"""
FLAT STRUCTURE-AWARE RAG PIPELINE
--------------------------------

Simplified architecture:
- NO parent-child hierarchy
- NO duplicate chunks
- Structure-aware chunking only
- Semantic-safe oversized splitting
- Rich metadata preserved

Pipeline:
PDF
↓
Extract structure
↓
Section/Subsection chunks
↓
Split oversized sections safely
↓
Embed all chunks directly
↓
Save JSON

Good for:
- manuals
- policy documents
- procurement guides
- government PDFs
"""

import re
import json
import fitz

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

PDF_PATH = "Works_Manual_SE_2025.pdf"
OUTPUT_PATH = "flat_chunks.json"

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

MAX_CHUNK_TOKENS = 350
SEMANTIC_THRESHOLD = 0.75

tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embedder = SentenceTransformer(EMBED_MODEL_NAME)


# -----------------------------------------------------------------------------
# TOKEN COUNT
# -----------------------------------------------------------------------------

def token_count(text):

    try:

        return len(
            tokenizer.encode(
                text,
                truncation=True,
                max_length=100000
            )
        )

    except:
        return 0


# -----------------------------------------------------------------------------
# CLEAN TEXT
# -----------------------------------------------------------------------------

def clean_text(text):

    # Remove page markers
    text = re.sub(r"<<<PAGE:\d+>>>", "", text)

    # Remove isolated page numbers
    text = re.sub(
        r"^\s*\d+\s*$",
        "",
        text,
        flags=re.MULTILINE
    )

    # Remove TOC dotted entries
    text = re.sub(
        r"^.*\.{4,}\s*\d+\s*$",
        "",
        text,
        flags=re.MULTILINE
    )

    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# -----------------------------------------------------------------------------
# PDF EXTRACTION
# -----------------------------------------------------------------------------

def extract_text_by_page(pdf_path):

    doc = fitz.open(pdf_path)

    pages = []

    for i, page in enumerate(doc):

        text = page.get_text("text")

        table_text = ""

        try:

            tables = page.find_tables()

            for table in tables:

                try:

                    extracted = table.extract()

                    rows = []

                    for row in extracted:

                        row = [
                            str(cell) if cell is not None else ""
                            for cell in row
                        ]

                        rows.append(" | ".join(row))

                    table_text += "\n" + "\n".join(rows)

                except:
                    pass

        except:
            pass

        combined = text + "\n" + table_text

        pages.append({
            "page": i + 1,
            "text": combined
        })

    return pages


def build_full_text(pages):

    return "\n".join(
        f"\n<<<PAGE:{p['page']}>>>\n{p['text']}"
        for p in pages
    )


# -----------------------------------------------------------------------------
# REGEX STRUCTURE
# -----------------------------------------------------------------------------

CHAPTER_RE = re.compile(
    r"^Chapter\s+(\d+)\s*[:\-\u2013]\s*(.+)$",
    re.IGNORECASE
)

SECTION_RE = re.compile(
    r"^(\d+(?:\.\d+)+)\s+(.+)$"
)

ANNEXURE_RE = re.compile(
    r"^(Annexure\s+[\dA-Z]+[:\-\u2013]?\s*.*)$",
    re.IGNORECASE
)

FALSE_SECTION_RE = re.compile(
    r"^\d+\.\d+.*?(of this manual|above|below|refer|see para)",
    re.IGNORECASE
)


# -----------------------------------------------------------------------------
# SEMANTIC BREAKS
# -----------------------------------------------------------------------------

def semantic_breaks(sentences, threshold=SEMANTIC_THRESHOLD):

    if len(sentences) <= 1:
        return []

    safe_sentences = []

    for s in sentences:

        try:

            tokens = tokenizer.encode(
                s,
                truncation=True,
                max_length=510
            )

            cleaned = tokenizer.decode(
                tokens,
                skip_special_tokens=True
            ).strip()

            if cleaned:
                safe_sentences.append(cleaned)

        except:
            continue

    if len(safe_sentences) <= 1:
        return []

    try:

        embeddings = embedder.encode(
            safe_sentences,
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True
        )

    except Exception as e:

        print(f"Embedding error: {e}")

        return []

    breaks = []

    for i in range(1, len(safe_sentences)):

        try:

            sim = cosine_similarity(
                [embeddings[i - 1]],
                [embeddings[i]]
            )[0][0]

            if sim < threshold:
                breaks.append(i)

        except:
            continue

    return breaks


# -----------------------------------------------------------------------------
# SPLIT LARGE SECTIONS
# -----------------------------------------------------------------------------

def split_large_chunk(text):

    text = clean_text(text)

    if token_count(text) <= MAX_CHUNK_TOKENS:
        return [text]

    # Sentence splitting
    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    # OCR fallback
    if len(sentences) <= 1:

        sentences = [
            s.strip()
            for s in text.split("\n")
            if s.strip()
        ]

    # Break giant malformed sentences
    cleaned_sentences = []

    for s in sentences:

        if token_count(s) > 500:

            mini_parts = re.split(r'[,;:\n]', s)

            cleaned_sentences.extend([
                x.strip()
                for x in mini_parts
                if x.strip()
            ])

        else:
            cleaned_sentences.append(s)

    sentences = cleaned_sentences

    semantic_points = semantic_breaks(sentences)

    chunks = []

    current = ""

    for idx, sentence in enumerate(sentences):

        if not sentence.strip():
            continue

        candidate = (
            current + " " + sentence
            if current else sentence
        )

        # Semantic split
        if idx in semantic_points and current.strip():

            chunks.append(current.strip())

            current = sentence

            continue

        # Token split
        if token_count(candidate) <= MAX_CHUNK_TOKENS:

            current = candidate

        else:

            if current.strip():
                chunks.append(current.strip())

            current = sentence

    if current.strip():
        chunks.append(current.strip())

    return [
        clean_text(c)
        for c in chunks
        if token_count(c) >= 15
    ]


# -----------------------------------------------------------------------------
# STRUCTURE PARSING
# -----------------------------------------------------------------------------

def parse_document_structure(full_text):

    chunks = []

    current_chapter = {
        "chapter_num": "",
        "chapter_title": ""
    }

    current_section = {
        "section_id": "",
        "section_title": ""
    }

    buffer = []

    page_hint = 1

    chunk_counter = 0

    # -------------------------------------------------------------------------
    # FLUSH
    # -------------------------------------------------------------------------

    def flush():

        nonlocal buffer
        nonlocal chunk_counter

        content = clean_text("\n".join(buffer))

        if token_count(content) < 20:

            buffer = []

            return

        split_chunks = split_large_chunk(content)

        for chunk_text in split_chunks:

            chunk = {
                "chunk_id": f"chunk_{chunk_counter:05d}",
                "chapter_num": current_chapter["chapter_num"],
                "chapter_title": current_chapter["chapter_title"],
                "section_id": current_section["section_id"],
                "section_title": current_section["section_title"],
                "page_hint": page_hint,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "token_count": token_count(chunk_text)
            }

            chunks.append(chunk)

            chunk_counter += 1

        buffer = []

    # -------------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------------

    lines = full_text.split("\n")

    for line in lines:

        pm = re.search(r"<<<PAGE:(\d+)>>>", line)

        if pm:

            page_hint = int(pm.group(1))

            continue

        stripped = line.strip()

        if not stripped:
            continue

        # ---------------------------------------------------------------------
        # CHAPTER
        # ---------------------------------------------------------------------

        cm = CHAPTER_RE.match(stripped)

        if cm:

            flush()

            current_chapter = {
                "chapter_num": cm.group(1),
                "chapter_title": cm.group(2).strip()
            }

            current_section = {
                "section_id": "",
                "section_title": f"Chapter {cm.group(1)} Overview"
            }

            buffer.append(stripped)

            continue

        # ---------------------------------------------------------------------
        # ANNEXURE
        # ---------------------------------------------------------------------

        am = ANNEXURE_RE.match(stripped)

        if am:

            flush()

            title = am.group(1).strip()

            current_section = {
                "section_id": "ANNEXURE",
                "section_title": title
            }

            buffer.append(stripped)

            continue

        # ---------------------------------------------------------------------
        # SECTION
        # ---------------------------------------------------------------------

        sm = SECTION_RE.match(stripped)

        if sm and not FALSE_SECTION_RE.match(stripped):

            flush()

            current_section = {
                "section_id": sm.group(1),
                "section_title": sm.group(2).strip()
            }

            buffer.append(stripped)

            continue

        # ---------------------------------------------------------------------
        # NORMAL CONTENT
        # ---------------------------------------------------------------------

        buffer.append(line)

    flush()

    return chunks


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def run_pipeline():

    print("\nExtracting PDF...")

    pages = extract_text_by_page(PDF_PATH)

    print(f"Pages extracted: {len(pages)}")

    full_text = build_full_text(pages)

    print("\nParsing document structure...")

    chunks = parse_document_structure(full_text)

    print(f"Final chunks created: {len(chunks)}")

    print("\nSaving JSON...")

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nSaved -> {OUTPUT_PATH}")

    print("\nSAMPLE OUTPUT")

    for chunk in chunks[:10]:

        print("\n" + "-" * 60)

        print(f"Chunk ID : {chunk['chunk_id']}")
        print(f"Chapter  : {chunk['chapter_num']}")
        print(f"Section  : {chunk['section_id']}")
        print(f"Title    : {chunk['section_title']}")
        print(f"Tokens   : {chunk['token_count']}")
        print(f"Page     : {chunk['page_hint']}")

        preview = chunk["text"][:200].replace("\n", " ")

        print(f"Preview  : {preview}...")

    return chunks


# -----------------------------------------------------------------------------

if __name__ == "__main__":

    run_pipeline()