"""
Advanced RAG Chunking Pipeline — Parent-Child Strategy
------------------------------------------------------
Features:
- Regex structural segmentation (Creates Parents)
- Tokenizer-aware chunking (tiktoken)
- Semantic/Sentence-aware splitting (Creates Children)
- Parent-child relational mapping
"""


import re
import json
import pdfplumber
import tiktoken
from pathlib import Path

# --- Configuration ---
PDF_PATH      = "/mnt/user-data/uploads/Works_Manual_SE_2025.pdf"
OUTPUT_PATH   = "chunks_parent_child.json"

CHILD_CHUNK_SIZE    = 200  # Target tokens for precision retrieval
CHILD_CHUNK_OVERLAP = 30   # Token overlap to preserve context between children

# Initialize tokenizer (standard OpenAI cl100k_base used by most embeddings)
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except ImportError:
    print("Please install tiktoken: pip install tiktoken")
    exit(1)

def count_tokens(text):
    return len(tokenizer.encode(text))

# --- Regex Patterns ---
CHAPTER_RE = re.compile(r"^Chapter\s+(\d+)\s*[:\-\u2013]\s*(.+)$", re.IGNORECASE)
SECTION_RE = re.compile(r"^(\d+\.\d+(?:\.\d+)?)\s+(.{4,90})$")
ANNEXURE_RE = re.compile(r"^(Annexure\s+[\dA-Z]+[:\-\u2013]?\s*.{0,60})$", re.IGNORECASE)
FALSE_SECTION_RE = re.compile(r"^\d+\.\d+.*?(of this manual|above|below|refer|see para)", re.IGNORECASE)

# --- 1. Extraction & Cleaning ---
def extract_text(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            pages.append(f"\n<<<PAGE:{i+1}>>>\n{page.extract_text() or ''}")
    return "\n".join(pages)

def clean_text(text):
    text = re.sub(r"<<<PAGE:\d+>>>", "", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE) # Remove isolated page numbers
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# --- 2. Parent Segmentation (The Context) ---
def build_parents(full_text):
    """Segments the document into logical sections (Parents) of ~600-900 tokens."""
    parents = []
    cur = dict(chapter_num="FM", chapter_title="Front Matter", section="00", section_title="Introduction", page_hint=1)
    buffer = []

    def flush_parent():
        content = clean_text("\n".join(buffer))
        if count_tokens(content) > 10:  # Ignore empty/tiny segments
            parent = dict(cur)
            parent["text"] = content
            parent["token_count"] = count_tokens(content)
            
            # Generate unique Parent ID
            ch = parent['chapter_num']
            sec = parent['section'].replace(".", "_")
            parent["parent_id"] = f"parent_ch{ch}_sec{sec}_{len(parents):04d}"
            
            parents.append(parent)
        buffer.clear()

    for line in full_text.split("\n"):
        pm = re.search(r"<<<PAGE:(\d+)>>>", line)
        if pm:
            cur["page_hint"] = int(pm.group(1))
            continue

        stripped = line.strip()

        # Match Chapter
        cm = CHAPTER_RE.match(stripped)
        if cm:
            flush_parent()
            cur.update(chapter_num=cm.group(1), chapter_title=cm.group(2).strip(),
                       section=cm.group(1), section_title=f"Chapter {cm.group(1)} Overview")
            buffer.append(line)
            continue

        # Match Section
        sm = SECTION_RE.match(stripped)
        if sm and not FALSE_SECTION_RE.match(stripped):
            sec_num = sm.group(1)
            # Ensure it belongs to the current chapter
            if sec_num.startswith(cur["chapter_num"] + "."):
                flush_parent()
                cur.update(section=sec_num, section_title=sm.group(2).strip())
                buffer.append(line)
                continue
                
        buffer.append(line)

    flush_parent()
    return parents

# --- 3. Child Sub-chunking (The Retrieval Targets) ---
def split_into_sentences(text):
    """Simple sentence splitter keeping delimiters."""
    return re.split(r'(?<=[.!?])\s+', text)

def build_children(parent):
    """Splits a parent segment into smaller semantic child chunks."""
    children = []
    parent_text = parent["text"]
    
    # If the parent is already small enough, it serves as its own child
    if parent["token_count"] <= CHILD_CHUNK_SIZE:
        child = dict(parent)
        child["chunk_id"] = f"child_{parent['parent_id']}_00"
        child["type"] = "child"
        return [child]

    # Split parent into paragraphs first, then sentences if paragraphs are too big
    paragraphs = parent_text.split("\n\n")
    semantic_blocks = []
    for p in paragraphs:
        if count_tokens(p) > CHILD_CHUNK_SIZE:
            semantic_blocks.extend(split_into_sentences(p))
        else:
            semantic_blocks.append(p)

    # Accumulate blocks into child chunks
    current_chunk_text = ""
    child_index = 0
    
    for block in semantic_blocks:
        candidate_text = current_chunk_text + ("\n\n" if current_chunk_text else "") + block
        
        if count_tokens(candidate_text) <= CHILD_CHUNK_SIZE:
            current_chunk_text = candidate_text
        else:
            # Flush current chunk
            if current_chunk_text:
                child = dict(parent) # Inherit metadata
                child["text"] = current_chunk_text.strip()
                child["token_count"] = count_tokens(child["text"])
                child["chunk_id"] = f"child_{parent['parent_id']}_{child_index:02d}"
                child["type"] = "child"
                children.append(child)
                child_index += 1
                
            # Start new chunk with overlap (take the last sentence of previous chunk)
            sentences = split_into_sentences(current_chunk_text)
            overlap_text = sentences[-1] if sentences else ""
            
            # If the block itself is massive, force split it
            if count_tokens(block) > CHILD_CHUNK_SIZE:
                current_chunk_text = block[:int(CHILD_CHUNK_SIZE * 4)] # fallback hard cut
            else:
                current_chunk_text = overlap_text + " " + block if overlap_text else block

    # Flush remaining
    if current_chunk_text.strip():
        child = dict(parent)
        child["text"] = current_chunk_text.strip()
        child["token_count"] = count_tokens(child["text"])
        child["chunk_id"] = f"child_{parent['parent_id']}_{child_index:02d}"
        child["type"] = "child"
        children.append(child)

    return children

# --- Main Pipeline Execution ---
def run_pipeline():
    print(" 1. Extracting text from PDF...")
    full_text = extract_text(PDF_PATH)
    
    print(" 2. Segmenting document into Parent Sections...")
    parents = build_parents(full_text)
    
    # Mark parents explicitly
    for p in parents:
        p["type"] = "parent"
        p["chunk_id"] = p["parent_id"] # for consistent primary key

    print("  3. Generating Semantic Child Chunks...")
    all_children = []
    for p in parents:
        all_children.extend(build_children(p))

    # Combine into a final schema structured for standard RAG ingestion
    output_data = {
        "metadata": {
            "source": PDF_PATH,
            "total_parents": len(parents),
            "total_children": len(all_children)
        },
        "parents": parents,
        "children": all_children
    }

    # Ensure directory exists
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n PIPELINE COMPLETE")
    print(f"  Parent Contexts (LLM Generation)  : {len(parents)}")
    print(f"  Child Chunks    (Vector Retrieval): {len(all_children)}")
    print(f" Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    run_pipeline()