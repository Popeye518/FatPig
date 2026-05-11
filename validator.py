#!/usr/bin/env python3
"""
SDLC Control Validator - Template JSON + Template PDF Evidence Validation
Validates solution design documentation using text embeddings and LLM-based quality assessment.
"""

import os
import io
import json
import argparse
import logging
import hashlib
import base64
import atexit
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Database & Connectors
import sqlalchemy
from sqlalchemy import text
from google.cloud.sql.connector import Connector, IPTypes

# Vertex AI (text embed LLM)
from google import genai
from google.genai import types as genai_types
import vertexai

# Auth for REST predict (multimodal embed)
import google.auth
from google.auth.transport.requests import Request as GARequest
import requests

# PDF/Docx parsing (Template PDF guidance text)
from PyPDF2 import PdfReader
from docx import Document as DocxDocument

# Local text chunking & cosine similarity for picking best template-PDF snippets
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

# PDF Generator
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Spacer


# ============================================================================
# CONFIG / CONSTANTS
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Vertex AI Configuration
PROJECT_ID = os.getenv("VERTEX_PROJECT_ID")
REGION = os.getenv("VERTEX_REGION", "europe-west3")

# Embedding & LLM Models
EMBED_MODEL = os.getenv("VERTEX_EMBED_MODEL", "text-embedding-005")
LLM_MODEL = os.getenv("VERTEX_LLM_MODEL", "gemini-2.5-flash")
MM_EMBED_MODEL = os.getenv("VERTEX_MM_EMBED_MODEL", "multimodalembedding@001")

# Embedding Dimensions
MM_DIM = int(os.getenv("VERTEX_MM_DIM", "1408"))

# Cloud SQL Connector Configuration
INSTANCE_CONNECTION_NAME = os.getenv("INSTANCE_CONNECTION_NAME", "")
PGUSER = os.getenv("PGUSER", os.getenv("PG_USER", "master"))
PGPASSWORD = os.getenv("PGPASSWORD", os.getenv("PG_PASSWORD", ""))
PGDATABASE = os.getenv("PGDATABASE", os.getenv("PG_DB", "database"))

# Database Table Names (aligned with your loaders & DDL)
TABLE_EVIDENCE = "vs_evidence"  # text evidence chunks
TABLE_GUIDANCE = "vs_guidance"  # guidance text chunks
TABLE_MM_EVID = "vs_mm_evidence_assets"  # evidence images

# Embedding Dimensions Map
EMBED_DIM_MAP: Dict[str, int] = {
    "text-embedding-005": 768,
    "gemini-embedding-001": 3072,
}
EMBED_DIM: Optional[int] = EMBED_DIM_MAP.get(EMBED_MODEL)

# Allowed Release Types
ALLOWED_RTYPES = {
    "indev": "New (indev) releases",
    "new": "GCP New releases",
    "migration": "GCP Migration releases",
    "rehost": "GCP Re-Host releases",
    "normal": "GCP Normal releases",
}


def resolve_rtype(user_val: str) -> str:
    """Resolve user-provided rtype to canonical form."""
    key = (user_val or "").strip().lower()
    if key in ALLOWED_RTYPES:
        return ALLOWED_RTYPES[key]
    for canonical in ALLOWED_RTYPES.values():
        if key == canonical.lower():
            return canonical
    raise ValueError("Invalid --rtype. Allowed: indev, new, migration, rehost, normal")


# ============================================================================
# Vertex AI Initialization
# ============================================================================

if not PROJECT_ID:
    logging.warning("VERTEX_PROJECT_ID is not set.")

vertexai.init(project=PROJECT_ID, location=REGION)
_genai = genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)


# ============================================================================
# Database: Cloud SQL Connector
# ============================================================================

_connector: Optional[Connector] = None
_engine: Optional[sqlalchemy.Engine] = None


def build_engine() -> sqlalchemy.Engine:
    """Build SQLAlchemy engine via Cloud SQL Connector."""
    global _connector
    _connector = Connector()

    def getconn():
        return _connector.connect(
            INSTANCE_CONNECTION_NAME,
            "pg8000",
            user=PGUSER,
            password=PGPASSWORD,
            db=PGDATABASE,
            ip_type=IPTypes.PSC,
        )

    return sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
        pool_pre_ping=True,
        pool_recycle=300,
    )


def get_engine() -> sqlalchemy.Engine:
    """Get or create the shared SQLAlchemy engine."""
    global _engine
    if _engine is None:
        if not INSTANCE_CONNECTION_NAME:
            raise RuntimeError("INSTANCE_CONNECTION_NAME is required for Cloud SQL Connector.")
        _engine = build_engine()
        logging.info("SQLAlchemy engine created via Cloud SQL Connector (pg8000).")
    return _engine


def _get_column_format_type(table_name: str, column_name: str) -> Optional[str]:
    """Return PostgreSQL's formatted column type, e.g. vector(1408)."""
    engine = get_engine()
    sql = text("""
        SELECT format_type(a.atttypid, a.atttypmod) AS format_type
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = :table_name
          AND a.attname = :column_name
          AND a.attnum > 0
          AND NOT a.attisdropped
        ORDER BY CASE WHEN n.nspname = current_schema() THEN 0 ELSE 1 END, n.nspname
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"table_name": table_name, "column_name": column_name}).first()
    return row[0] if row and row[0] else None


def _parse_vector_dimension(format_type_name: Optional[str]) -> Optional[int]:
    """Parse a vector(N) type string into its dimension."""
    if not format_type_name:
        return None
    match = re.fullmatch(r"vector\((\d+)\)", format_type_name.strip(), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def validate_runtime_contract(mm_dim: int) -> None:
    """Fail fast when runtime configuration does not match the database contract."""
    if EMBED_DIM is not None:
        text_embedding_type = _get_column_format_type(TABLE_EVIDENCE, "embedding")
        text_embedding_dim = _parse_vector_dimension(text_embedding_type)
        if text_embedding_dim != EMBED_DIM:
            raise RuntimeError(
                f"{TABLE_EVIDENCE}.embedding expects {text_embedding_type or 'unknown'}, "
                f"but {EMBED_MODEL} is configured for dimension {EMBED_DIM}."
            )

    guidance_embedding_type = _get_column_format_type(TABLE_GUIDANCE, "embedding")
    guidance_embedding_dim = _parse_vector_dimension(guidance_embedding_type)
    if guidance_embedding_dim is not None and EMBED_DIM is not None and guidance_embedding_dim != EMBED_DIM:
        raise RuntimeError(
            f"{TABLE_GUIDANCE}.embedding expects {guidance_embedding_type}, "
            f"but {EMBED_MODEL} is configured for dimension {EMBED_DIM}."
        )

    mm_embedding_type = _get_column_format_type(TABLE_MM_EVID, "embedding")
    mm_embedding_dim = _parse_vector_dimension(mm_embedding_type)
    if mm_embedding_dim != mm_dim:
        raise RuntimeError(
            f"{TABLE_MM_EVID}.embedding expects {mm_embedding_type or 'unknown'}, "
            f"but VERTEX_MM_DIM is configured for dimension {mm_dim}."
        )


def shutdown_connector():
    """Cleanup: close Cloud SQL connector."""
    global _connector, _engine
    try:
        if _engine is not None:
            _engine.dispose()
    finally:
        if _connector is not None:
            _connector.close()
        logging.info("Cloud SQL Connector closed.")


atexit.register(shutdown_connector)


# ============================================================================
# Embedding & LLM Helpers
# ============================================================================

def _extract_values(resp) -> List[float]:
    """Extract embedding vector from various response formats."""
    if hasattr(resp, "embedding") and hasattr(resp.embedding, "values"):
        return resp.embedding.values
    if hasattr(resp, "values"):
        return resp.values
    if hasattr(resp, "embeddings") and resp.embeddings:
        for e in resp.embeddings:
            if hasattr(e, "values"):
                return e.values
            if isinstance(e, dict) and "values" in e:
                return e["values"]
    if isinstance(resp, dict):
        if "embedding" in resp and isinstance(resp["embedding"], dict) and "values" in resp["embedding"]:
            return resp["embedding"]["values"]
        if "values" in resp:
            return resp["values"]
    raise RuntimeError(f"Unexpected embed response shape: {type(resp)}")


def embed_texts(
    texts: List[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    out_dim: Optional[int] = EMBED_DIM,
) -> List[List[float]]:
    """Embed multiple text chunks."""
    cfg = genai_types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=out_dim,
    )
    out = []
    for t in texts:
        r = _genai.models.embed_content(model=EMBED_MODEL, contents=t, config=cfg)
        out.append(_extract_values(r))
    return out


def embed_query_text(
    text_q: str,
    out_dim: Optional[int] = EMBED_DIM,
) -> List[float]:
    """Embed a query text."""
    cfg = genai_types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=out_dim,
    )
    r = _genai.models.embed_content(model=EMBED_MODEL, contents=text_q, config=cfg)
    return _extract_values(r)


def get_access_token() -> str:
    """Get Google Cloud access token for REST API calls."""
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    if not creds.valid:
        creds.refresh(GARequest())
    return creds.token


def mm_embed_text(query: str, dim: int = MM_DIM) -> List[float]:
    """Embed text using multimodal embedding model (REST API)."""
    token = get_access_token()
    endpoint = (
        f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}"
        f"/publishers/google/models/{MM_EMBED_MODEL}:predict"
    )
    payload = {
        "instances": [
            {
                "text": query,
                "parameters": {"dimension": dim},
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    preds = data.get("predictions") or []
    if not preds or "textEmbedding" not in preds[0]:
        raise RuntimeError(f"Vertex MM: expected textEmbedding. data_keys={list(data.keys())}")
    vec = preds[0]["textEmbedding"]
    if dim and len(vec) != dim:
        raise RuntimeError(f"Vertex MM: expected dim={dim}, got len={len(vec)}")
    return vec


def mm_embed_image_with_caption(
    img_bytes: bytes,
    caption: str,
    dim: int = MM_DIM,
) -> List[float]:
    """Embed image with optional caption using multimodal embedding model."""
    token = get_access_token()
    endpoint = (
        f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}"
        f"/publishers/google/models/{MM_EMBED_MODEL}:predict"
    )
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    instance = {
        "image": {
            "bytesBase64Encoded": b64,
            "mimeType": "image/png",
        },
        "parameters": {"dimension": dim},
    }
    if caption:
        instance["text"] = caption[:200]
    payload = {"instances": [instance]}
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    preds = data.get("predictions") or []
    if not preds:
        raise RuntimeError("Vertex MM: missing predictions")
    pred = preds[0]
    vec = pred.get("imageEmbedding") or pred.get("textEmbedding")
    if vec is None:
        raise RuntimeError(f"Vertex MM: missing embedding. keys={list(pred.keys())}")
    if dim and len(vec) != dim:
        raise RuntimeError(f"Vertex MM: expected dim={dim}, got len={len(vec)}")
    return vec


def gen_text(prompt: str) -> str:
    """Generate text using LLM."""
    try:
        generation_config = genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=2192,
        )
        resp = _genai.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=generation_config,
        )
        return (
            getattr(resp, "output_text", None)
            or getattr(resp, "text", "")
            or ""
        )
    except Exception as e:
        logging.error(f"LLM generation failed: {e}")
        return ""


def parse_json(s: str) -> Dict[str, Any]:
    """Parse JSON from string, handling markdown code blocks."""
    s = (s or "").strip()
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()
    try:
        return json.loads(s)
    except Exception as e:
        logging.warning(f"parse_json failed: {e} | raw {s[:200]}")
        return {}


def parse_json_with_error(s: str) -> Tuple[Dict[str, Any], Optional[str], str]:
    """Parse JSON and preserve parser failures as structured errors."""
    cleaned = (s or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    if not cleaned:
        return {}, "empty model response", cleaned
    try:
        parsed = json.loads(cleaned)
    except Exception as exc:
        return {}, f"JSON parse failed: {exc}", cleaned
    if not isinstance(parsed, dict):
        return {}, f"Expected JSON object, got {type(parsed).__name__}", cleaned
    return parsed, None, cleaned


def generate_structured_json(prompt: str, response_kind: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """Generate structured JSON with one repair retry when parsing fails."""
    raw_response = gen_text(prompt)
    parsed, error, cleaned = parse_json_with_error(raw_response)
    if error is None:
        return parsed, None

    logging.warning(f"{response_kind} JSON parse failed on first attempt: {error}")

    repair_prompt = (
        f"{prompt}\n\n"
        "Your previous response was not valid JSON. "
        "Return STRICT JSON ONLY that matches the requested schema and do not include markdown fences.\n\n"
        "Previous invalid response:\n"
        f"{cleaned[:4000]}"
    )
    repaired_response = gen_text(repair_prompt)
    repaired, repair_error, repaired_cleaned = parse_json_with_error(repaired_response)
    if repair_error is None:
        return repaired, None

    combined_error = f"{error}; retry failed: {repair_error}"
    logging.error(f"{response_kind} JSON parse failed after retry: {combined_error}")
    return {
        "status": "ERROR",
        "quality": {},
        "justification": (
            f"Validator could not parse model output for {response_kind}. "
            "Review logs and retrieved evidence context."
        ),
        "citations": {},
        "validator_error": combined_error,
        "raw_response_excerpt": repaired_cleaned[:500],
    }, combined_error


# ============================================================================
# Document Processing
# ============================================================================

def read_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF file."""
    reader = PdfReader(pdf_path)
    text = "\n".join([(pg.extract_text() or "") for pg in reader.pages])
    return text


def read_docx_text(docx_path: str) -> str:
    """Extract text from DOCX file."""
    doc = DocxDocument(docx_path)
    return "\n".join([p.text for p in doc.paragraphs])


def compute_doc_hash(file_bytes: bytes) -> str:
    """Compute SHA-256 hash of file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def split_text_local(
    text_in: str,
    chunk_size: int = 900,
    chunk_overlap: int = 90,
) -> List[str]:
    """Split text into chunks locally."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text_in)


def semantic_tag_text(tag: str) -> str:
    """Extract semantic content from tag (remove numbering)."""
    text = (tag or "").strip()
    # Remove leading numbers like "1.2.3"
    text = re.sub(r"^\s*\d+(?:[.,]\d+)*(?:\.[a-z])?\s+", "", text, flags=re.IGNORECASE)
    # Remove leading section labels like "A1", "B2.3"
    text = re.sub(r"^\s*[A-Za-z]\d+(?:[.,]\d+)*(?:\.[a-z])?\s+", "", text, flags=re.IGNORECASE)
    return text.strip() or (tag or "").strip()


def extract_tag_prefix(tag: str) -> str:
    """Extract numbering prefix from tag (e.g., '1.2' from '1.2 Title')."""
    match = re.match(
        r"^\s*([A-Za-z]?\d+(?:[.,]\d+)*(?:\.[a-z])?)\b",
        (tag or "").strip(),
        flags=re.IGNORECASE,
    )
    return match.group(1).strip().lower() if match else ""


def _normalize_match_text(text_in: str) -> str:
    """Normalize text for matching: lowercase, remove non-alphanumeric."""
    return re.sub(
        r"\s+",
        "",
        re.sub(r"[^a-z0-9]+", "", (text_in or "").lower()),
    ).strip()


def _normalize_line_text(text_in: str) -> str:
    """Normalize text while preserving word boundaries for line-based matching."""
    return re.sub(
        r"\s+",
        " ",
        re.sub(r"[^a-z0-9\s]+", " ", (text_in or "").lower()),
    ).strip()


def _normalize_multiline_text(text_in: str) -> str:
    """Normalize text line by line while preserving line boundaries."""
    lines = []
    for raw_line in re.split(r"\r?\n", text_in or ""):
        normalized_line = _normalize_line_text(raw_line)
        if normalized_line:
            lines.append(normalized_line)
    return "\n".join(lines)


def _tokenize_match_text(text_in: str) -> List[str]:
    """Tokenize text for loose semantic overlap checks."""
    return re.findall(r"[a-z0-9]+", (text_in or "").lower())


def _load_template_nodes(template_json_path: str) -> List[Dict[str, Any]]:
    """Load template nodes from either a top-level list or an object with children."""
    with open(template_json_path, "r", encoding="utf-8") as f:
        template = json.load(f)

    if isinstance(template, list):
        return [node for node in template if isinstance(node, dict)]

    if isinstance(template, dict):
        children = template.get("children")
        if isinstance(children, list):
            return [node for node in children if isinstance(node, dict)]

    raise ValueError(
        "Template JSON must be either a list of intent nodes or an object with a 'children' list."
    )


def topic_alignment_status(
    tag: str,
    semantic_tag: str,
    evidence_snippets: List[str],
) -> Optional[str]:
    """Check if topic is aligned with expected template tag."""
    tag_prefix = extract_tag_prefix(tag)
    if not tag_prefix or not evidence_snippets:
        return None
    normalized_topic = _normalize_line_text(semantic_tag)
    if not normalized_topic:
        return None
    tag_prefix_pattern = re.escape(tag_prefix).replace(r"\.", r"[.\s]+")
    topic_pattern = re.escape(normalized_topic).replace(r"\ ", r"\s+")
    aligned_pattern = re.compile(
        rf"(^|[\r\n])\s*({tag_prefix_pattern})(?:[\s:\-]+)({topic_pattern})\b",
        flags=re.IGNORECASE,
    )
    topic_line_pattern = re.compile(
        rf"(^|[\r\n])\s*(?:[A-Za-z]?\d+(?:[.\s]+\d+)*(?:\.[a-z])?)(?:[\s:\-]+){topic_pattern}\b",
        flags=re.IGNORECASE,
    )
    normalized_joined = "\n".join(
        _normalize_multiline_text(snippet) for snippet in evidence_snippets if snippet
    )
    if not normalized_joined.strip():
        return None
    if aligned_pattern.search(normalized_joined):
        return "aligned"
    if topic_line_pattern.search(normalized_joined):
        return "misaligned"
    return None


def build_topic_queries(
    intent_name: str,
    semantic_tag: str,
    tag: str,
    notes: str,
    conds: str,
    kw_text: str,
) -> List[str]:
    """Build multiple query strings for topic retrieval."""
    candidates = [
        " ".join(s for s in [semantic_tag, kw_text] if s).strip(),
        " ".join(s for s in [intent_name, semantic_tag, kw_text] if s).strip(),
        " ".join(s for s in [semantic_tag, notes] if s).strip(),
        " ".join(s for s in [semantic_tag, conds] if s).strip(),
        " ".join(s for s in [tag, semantic_tag, notes, conds, kw_text] if s).strip(),
    ]
    queries: List[str] = []
    seen = set()
    for candidate in candidates:
        normalized = _normalize_match_text(candidate)
        if normalized and normalized not in seen:
            seen.add(normalized)
            queries.append(candidate)
    return queries


def build_topic_search_phrases(
    semantic_tag: str,
    tag: str,
    keywords: List[str],
) -> List[str]:
    """Build search phrases for keyword-based retrieval."""
    candidates = [tag, semantic_tag] + [k for k in keywords if isinstance(k, str)]
    phrases: List[str] = []
    seen = set()
    for candidate in candidates:
        raw = (candidate or "").strip()
        normalized = _normalize_match_text(candidate)
        if raw and normalized and len(normalized) > 4 and normalized not in seen:
            seen.add(normalized)
            phrases.append(raw)
    return phrases


def retrieve_keyword_evidence_snippets(
    phrases: List[str],
    nar_id: str,
    release_number: str,
    rtype: str,
    doc_id: Optional[str],
    top_n: int,
) -> List[str]:
    """Retrieve evidence snippets using keyword matching."""
    valid_phrases = []
    seen = set()
    for phrase in phrases:
        normalized = _normalize_match_text(phrase)
        if normalized and normalized not in seen:
            seen.add(normalized)
            valid_phrases.append((phrase, normalized))
    if not valid_phrases:
        return []
    where = ["nar_id=:nar", "release_number=:rel", "rtype=:rtype"]
    params: Dict[str, Any] = {
        "nar": nar_id,
        "rel": release_number,
        "rtype": rtype,
        "topn": top_n,
    }
    if doc_id:
        where.append("doc_id=:doc_id")
        params["doc_id"] = doc_id
    phrase_conditions = []
    exact_checks = []
    for idx, (raw_phrase, normalized_phrase) in enumerate(valid_phrases):
        like_key = f"phrase_{idx}"
        exact_key = f"exact_{idx}"
        params[like_key] = f"%{raw_phrase}%"
        params[exact_key] = f"%{normalized_phrase}%"
        phrase_conditions.append(f"chunk ILIKE :{like_key}")
        exact_checks.append(
            f"CASE WHEN lower(regexp_replace(chunk, '[^a-zA-Z0-9]', '', 'g')) LIKE :{exact_key} THEN 0 ELSE 1 END"
        )
    sql = text(f"""
        SELECT chunk
        FROM {TABLE_EVIDENCE}
        WHERE {" AND ".join(where)}
          AND ({" OR ".join(phrase_conditions)})
        ORDER BY {", ".join(exact_checks)}, length(chunk)
        LIMIT :topn
    """)
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [r[0] for r in rows if r and (r[0] or "").strip()]


def retrieve_topic_evidence_snippets(
    queries: List[str],
    nar_id: str,
    release_number: str,
    rtype: str,
    doc_id: Optional[str],
    top_n: int,
    search_phrases: Optional[List[str]] = None,
) -> List[str]:
    """Retrieve evidence snippets using semantic + keyword search."""
    merged: List[str] = []
    per_query_top_n = max(3, min(top_n, 4))
    seen = set()
    if search_phrases:
        lexical_hits = retrieve_keyword_evidence_snippets(
            search_phrases, nar_id, release_number, rtype, doc_id, top_n
        )
        for snippet in lexical_hits:
            normalized = _normalize_match_text(snippet)
            if normalized and normalized not in seen:
                seen.add(normalized)
                merged.append(snippet)
        if len(merged) >= top_n:
            return merged
    for query in queries:
        snippets = retrieve_evidence_snippets(
            query, nar_id, release_number, rtype, doc_id, per_query_top_n
        )
        for snippet in snippets:
            normalized = _normalize_match_text(snippet)
            if normalized and normalized not in seen:
                seen.add(normalized)
                merged.append(snippet)
        if len(merged) >= top_n:
            return merged
    return merged


def semantic_topic_match_strength(
    semantic_tag: str,
    keywords: List[str],
    evidence_snippets: List[str],
) -> Optional[str]:
    """Assess semantic match strength: 'exact', 'partial', or None."""
    if not evidence_snippets:
        return None
    joined = _normalize_match_text(" ".join(evidence_snippets))
    if not joined:
        return None
    candidate_phrases = [semantic_tag] + [k for k in keywords if isinstance(k, str)]
    normalized_phrases = []
    for phrase in candidate_phrases:
        normalized = _normalize_match_text(phrase)
        if normalized and len(normalized) >= 4:
            normalized_phrases.append(normalized)
    if any(phrase in joined for phrase in normalized_phrases):
        return "exact"
    stop_words = {
        "and", "the", "for", "with", "from", "into", "that", "this",
        "have", "must", "good", "include", "table", "section", "application",
        "overview", "details", "link",
    }
    evidence_token_set = set(_tokenize_match_text(" ".join(evidence_snippets)))
    semantic_tokens = [
        token
        for token in _tokenize_match_text(semantic_tag)
        if len(token) > 4 and token not in stop_words
    ]
    if not semantic_tokens:
        return None
    token_hits = sum(
        1
        for token in semantic_tokens
        if token in evidence_token_set
    )
    required_hits = min(len(semantic_tokens), max(2, (len(semantic_tokens) + 1) // 2))
    return "partial" if token_hits >= required_hits else None


def pick_best_template_snippets_for_tag(
    tag_query: str,
    pdf_text: str,
    top_n: int = 3,
) -> List[str]:
    """Pick best matching snippets from template PDF using embeddings."""
    if not pdf_text.strip():
        return []
    chunks = split_text_local(pdf_text)
    if not chunks:
        return []
    emb_chunks = embed_texts(chunks, task_type="RETRIEVAL_DOCUMENT", out_dim=EMBED_DIM)
    qv = np.array(embed_query_text(tag_query, out_dim=EMBED_DIM), dtype=np.float32).reshape(1, -1)
    em = np.array(emb_chunks, dtype=np.float32)
    sims = cosine_similarity(qv, em)[0]
    idx = np.argsort(-sims)[:top_n]
    return [chunks[i] for i in idx]


# ============================================================================
# Vector Literal & Retrieval Functions
# ============================================================================

def _vec_literal(vec: List[float]) -> str:
    """Format vector as PostgreSQL pgvector literal."""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def retrieve_evidence_snippets(
    query: str,
    nar_id: str,
    release_number: str,
    rtype: str,
    doc_id: Optional[str],
    top_n: int,
) -> List[str]:
    """Retrieve evidence snippets using semantic search."""
    q_emb = embed_query_text(query)
    vec = _vec_literal(q_emb)
    where = ["nar_id=:nar", "release_number=:rel", "rtype=:rtype"]
    params = {
        "nar": nar_id,
        "rel": release_number,
        "rtype": rtype,
        "vec": vec,
        "topn": top_n,
    }
    if doc_id:
        where.append("doc_id=:doc_id")
        params["doc_id"] = doc_id
    sql = text(f"""
        SELECT chunk
        FROM {TABLE_EVIDENCE}
        WHERE {" AND ".join(where)}
        ORDER BY embedding <-> CAST(:vec AS vector)
        LIMIT :topn
    """)
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [r[0] for r in rows if r and (r[0] or "").strip()]


def retrieve_guidance_snippets(
    query: str,
    rtype: str,
    top_n: int = 3,
) -> List[str]:
    """Retrieve guidance snippets."""
    try:
        qv = embed_query_text(query)
        vec = _vec_literal(qv)
        sql = text(f"""
            SELECT chunk
            FROM {TABLE_GUIDANCE}
            WHERE rtype=:rtype
            ORDER BY embedding <-> CAST(:vec AS vector)
            LIMIT :topn
        """)
        engine = get_engine()
        with engine.connect() as c:
            rows = c.execute(sql, {"rtype": rtype, "vec": vec, "topn": top_n}).fetchall()
        return [r[0] for r in rows if r and r[0]]
    except Exception as e:
        logging.warning(f"guidance retrieval failed: {e}")
        return []


def retrieve_mm_diagrams(
    query_text: str,
    nar_id: str,
    release_number: str,
    rtype: str,
    doc_id: Optional[str],
    top_n: int = 3,
    dim: int = MM_DIM,
) -> List[Dict[str, str]]:
    """Retrieve multimodal diagram references."""
    qv = mm_embed_text(query_text, dim)
    vec = _vec_literal(qv)
    where = ["nar_id=:nar", "release_number=:rel", "rtype=:rtype"]
    params: Dict[str, Any] = {
        "nar": nar_id,
        "rel": release_number,
        "rtype": rtype,
        "vec": vec,
        "topn": top_n,
    }
    if doc_id:
        where.append("doc_id=:doc_id")
        params["doc_id"] = doc_id
    sql = text(f"""
        SELECT caption, doc_uri
        FROM {TABLE_MM_EVID}
        WHERE {" AND ".join(where)}
        ORDER BY embedding <-> CAST(:vec AS vector)
        LIMIT :topn
    """)
    engine = get_engine()
    with engine.connect() as c:
        rows = c.execute(sql, params).fetchall()
    return [{"caption": r[0], "doc_uri": r[1]} for r in rows]


# ============================================================================
# LLM PROMPTS
# ============================================================================

PRESENCE_QUALITY_PROMPT = """You are a validation agent.

Decide PRESENCE and score QUALITY using:
- Template tag & notes
- Template-PDF criteria excerpts (authoritative instructions)
- Optional Guidance snippets (org guidance)
- Evidence text snippets

Return STRICT JSON only.

Important evaluation rule:
Judge semantic content, not exact section numbering.
If the right content appears under a different numbering or heading label (for example 1.2 instead of 1.1), do NOT mark it PARTIAL or MISSING for that reason alone.
Only mark PARTIAL or MISSING when required content itself is incomplete, generic, contradictory, or absent.

TAG: {tag}
INTENT: {intent}
SEMANTIC TAG: {semantic_tag}
NOTES: {notes}
REQUIRED: {required_flag}

TEMPLATE PDF EXCERPTS:
{template_pdf_excerpts}

GUIDANCE SNIPPETS:
{guidance}

EVIDENCE SNIPPETS:
{evidence}

Respond JSON:
{{
  "status": "PRESENT" | "PARTIAL" | "MISSING" | "EMPTY",
  "quality": {{
    "completeness": 0-5,
    "specificity": 0-5,
    "traceability": 0-5
  }},
  "justification": "short one or two sentences",
  "citations": {{
    "evidence": ["short excerpt 1", "short excerpt 2"],
    "guidance": ["short excerpt or empty"],
    "template_pdf": ["short excerpt 1"]
  }}
}}
"""

DIAGRAM_PROMPT = """You are validating whether the required DIAGRAM exists and is of good quality.

Use Template tag/notes, Template-PDF criteria, optional Guidance, and DIAGRAM REFERENCES.
Return STRICT JSON only.

Important evaluation rule:
Judge semantic alignment, not exact section numbering.
If the diagram/content is present under a different numbered section, do NOT penalize that numbering mismatch by itself.

TAG: {tag}
INTENT: {intent}
SEMANTIC TAG: {semantic_tag}
NOTES: {notes}
REQUIRED: {required_flag}

TEMPLATE_PDF_EXCERPTS:
{template_pdf_excerpts}

GUIDANCE SNIPPETS:
{guidance}

EVIDENCE TEXT SNIPPETS:
{evidence}

DIAGRAM_REFERENCES (top-k by similarity):
{diagrams}

Respond JSON:
{{
  "status": "PRESENT" | "PARTIAL" | "MISSING" | "EMPTY",
  "quality": {{
    "completeness": 0-5,
    "specificity": 0-5,
    "diagram_alignment": 0-5
  }},
  "justification": "short one or two sentences",
  "citations": {{
    "diagram": ["<caption or uri>"],
    "text": ["short excerpt or empty"],
    "template_pdf": ["short excerpt 1"]
  }}
}}
"""

ARCHITECTURE_SUMMARY_PROMPT = """Summarize architecture coverage and quality (conceptual, logical, physical).
Use text evidence snippets below (top-k) plus Template-PDF expectations.

TEMPLATE_PDF_EXCERPTS:
{template_pdf_excerpts}

EVIDENCE SNIPPETS:
{evidence}

Return STRICT JSON:
{{
  "summary": "concise summary",
  "conceptual_present": true | false,
  "logical_present": true | false,
  "physical_present": true | false,
  "label_alignment_issues": ["example or empty"],
  "recommendations": ["short, actionable rec #1", "short, actionable rec #2"]
}}
"""


# ============================================================================
# Core Validation Engine
# ============================================================================

def run_validation(
    template_json_path: str,
    template_pdf_path: str,
    nar_id: str,
    application_name: str,
    release_number: str,
    rtype: str,
    top_k: int = 6,
    scope_doc_id: Optional[str] = None,
    scope_doc_hash: Optional[str] = None,
    enable_mm: bool = True,
) -> Dict[str, Any]:
    """Run full validation workflow."""
    effective_doc_id = (scope_doc_id or scope_doc_hash or "").strip() or None
    if scope_doc_id and scope_doc_hash and scope_doc_id != scope_doc_hash:
        raise ValueError("scope_doc_id and scope_doc_hash must match when both are provided.")

    if EMBED_DIM is None:
        logging.warning(
            f"Unknown embedding model '{EMBED_MODEL}'. "
            f"Ensure VECTOR(dim) matches model output."
        )
    
    template_nodes = _load_template_nodes(template_json_path)
    
    # Extract Template PDF text
    if template_pdf_path.lower().endswith(".pdf"):
        pdf_text = read_pdf_text(template_pdf_path)
    elif template_pdf_path.lower().endswith(".docx"):
        pdf_text = read_docx_text(template_pdf_path)
    else:
        logging.warning("Template PDF path is not PDF/DOCX; skipping PDF guidance extraction.")
        pdf_text = ""
    
    results: List[Dict[str, Any]] = []
    must_total = 0
    must_met = 0
    good_total = 0
    good_met = 0
    
    # Iterate over template
    for node in template_nodes:
        intent_name = (node.get("intent") or "").strip()
        for child in node.get("children", []):
            tag = (child.get("tag") or "").strip()
            semantic_tag = semantic_tag_text(tag)
            mreq = bool(child.get("musthave", False))
            greq = bool(child.get("goodToHave", False))
            notes = (child.get("notes", "") or "").strip()
            conds = (child.get("conditions", "") or "").strip()
            kws = child.get("keywords") or []
            kw_text = " ".join(k.strip() for k in kws if isinstance(k, str) and k.strip())
            
            # Topic-first retrieval queries
            topic_queries = build_topic_queries(intent_name, semantic_tag, tag, notes, conds, kw_text)
            topic_search_phrases = build_topic_search_phrases(semantic_tag, tag, kws)
            query_full = topic_queries[0] if topic_queries else " ".join(s for s in [semantic_tag, kw_text] if s).strip()
            
            # Evidence text retrieval
            ev_snips = retrieve_topic_evidence_snippets(
                topic_queries,
                nar_id,
                release_number,
                rtype,
                effective_doc_id,
                top_k,
                topic_search_phrases,
            )
            s_join = "\n".join(ev_snips[:top_k]) if ev_snips else "(no snippets found)"
            
            # Guidance retrieval (optional)
            g_snips = retrieve_guidance_snippets(query_full, rtype, top_n=3)
            g_join = "\n".join(g_snips) if g_snips else "(no guidance context)"
            
            # Template PDF excerpts relevant to this tag
            pdf_excerpts = pick_best_template_snippets_for_tag(query_full, pdf_text, top_n=3)
            pdf_join = "\n".join(pdf_excerpts) if pdf_excerpts else "(no template-pdf excerpt)"
            
            # Diagram detection
            is_diagram = (
                ("diagram" in tag.lower())
                or ("diagram" in notes.lower())
                or ("diagram" in conds.lower())
                or any(isinstance(k, str) and "diagram" in k.lower() for k in kws)
            )
            
            if enable_mm and is_diagram:
                mm_hits = retrieve_mm_diagrams(
                    query_full,
                    nar_id,
                    release_number,
                    rtype,
                    effective_doc_id,
                    top_n=3,
                    dim=MM_DIM,
                )
                diag_block = "\n".join([f"{h['caption']}:: {h['doc_uri']}" for h in mm_hits]) or "(none)"
                p_json, validation_error = generate_structured_json(
                    DIAGRAM_PROMPT.format(
                        tag=tag,
                        intent=intent_name,
                        semantic_tag=semantic_tag,
                        notes=notes,
                        required_flag=(
                            "MUST_HAVE" if mreq else ("GOOD_TO_HAVE" if greq else "OPTIONAL")
                        ),
                        template_pdf_excerpts=pdf_join,
                        guidance=g_join,
                        evidence=s_join,
                        diagrams=diag_block,
                    ),
                    response_kind=f"diagram validation for tag '{tag}'",
                )
            else:
                mm_hits = []
                p_json, validation_error = generate_structured_json(
                    PRESENCE_QUALITY_PROMPT.format(
                        tag=tag,
                        intent=intent_name,
                        semantic_tag=semantic_tag,
                        notes=notes,
                        required_flag=(
                            "MUST_HAVE" if mreq else ("GOOD_TO_HAVE" if greq else "OPTIONAL")
                        ),
                        template_pdf_excerpts=pdf_join,
                        guidance=g_join,
                        evidence=s_join,
                    ),
                    response_kind=f"presence validation for tag '{tag}'",
                )
            
            status = ((p_json.get("status") if isinstance(p_json, dict) else None) or "MISSING").upper()
            justification = p_json.get("justification", "") if isinstance(p_json, dict) else ""
            
            tag_1 = (tag or "").strip().lower()
            topic_match = semantic_topic_match_strength(semantic_tag, kws, ev_snips)
            topic_alignment = topic_alignment_status(tag, semantic_tag, ev_snips)
            
            if topic_alignment == "aligned" and topic_match in ("exact", "partial") and status in ("MISSING", "PARTIAL"):
                status = "PRESENT"
                topic_note = (
                    f"Topic '{semantic_tag}' is present under the expected template tag '{tag}'. "
                    "Marked PRESENT because the section alignment matches the template."
                )
                justification = (
                    f"{justification} {topic_note}".strip() if justification else topic_note
                )
            elif topic_match in ("exact", "partial") and (topic_alignment == "misaligned" or status == "MISSING"):
                status = "PARTIAL"
                topic_note = (
                    f"Topic '{semantic_tag}' appears in the evidence, but not under the expected template tag '{tag}'. "
                    "Marked PARTIAL because the content is present but the section/tag alignment does not match the template."
                )
                justification = (
                    f"{justification} {topic_note}".strip() if justification else topic_note
                )
            
            if tag_1 in ("0.1 nar id", "8.1 nar id", "nar id") and nar_id:
                status = "PRESENT"
                justification = f"NAR ID '{nar_id}' was resolved from evidence."
            
            if tag_1 in ("0.2 app name", "0.2 application name", "app name", "application name") and application_name:
                status = "PRESENT"
                justification = f"Application name '{application_name}' was resolved from evidence."

            if tag_1 in ("0.3 r-type", "0.3 r type", "r-type", "r type", "rtype") and rtype:
                status = "PRESENT"
                justification = f"R-Type '{rtype}' was provided for validation."
            
            if mreq:
                must_total += 1
                if status == "PRESENT":
                    must_met += 1
            
            if greq:
                good_total += 1
                if status == "PRESENT":
                    good_met += 1
            
            results.append({
                "intent": intent_name,
                "tag": tag,
                "musthave": mreq,
                "good_to_have": greq,
                "status": status,
                "quality": p_json.get("quality", {}) if isinstance(p_json, dict) else {},
                "justification": justification,
                "citations": p_json.get("citations", {}) if isinstance(p_json, dict) else {},
                "diagram_refs": mm_hits,
                "snippets": ev_snips[:top_k],
                "used_guidance": bool(g_snips),
                "template_pdf_used": bool(pdf_excerpts),
                "validator_error": validation_error,
            })
    
    # Architecture summary (conceptual, logical, physical)
    arch_query = "overall Architecture Conceptual Logical Physical GKE Cloud SQL Interfaces components"
    arch_snips = retrieve_evidence_snippets(
        arch_query, nar_id, release_number, rtype, effective_doc_id, top_n=18
    )
    arch_s_join = "\n".join(arch_snips) if arch_snips else "(none)"
    arch_pdf = pick_best_template_snippets_for_tag(
        "Overall Architecture Diagrams Conceptual Logical Physical",
        pdf_text,
        top_n=3,
    )
    arch_pdf_join = "\n".join(arch_pdf) if arch_pdf else "(no template-pdf excerpt)"
    
    arch_json, architecture_error = generate_structured_json(
        ARCHITECTURE_SUMMARY_PROMPT.format(
            template_pdf_excerpts=arch_pdf_join,
            evidence=arch_s_join,
        ),
        response_kind="architecture summary",
    )
    
    logging.info(f"arch_json result: {arch_json}")
    
    summary = {
        "nar_id": nar_id,
        "application_name": application_name,
        "release_number": release_number,
        "rtype": rtype,
        "doc_id_scope": effective_doc_id,
        "doc_hash_scope": effective_doc_id,
        "must_have_total": must_total,
        "must_have_coverage_pct": round((must_met / must_total * 100.0), 1) if must_total else 0.0,
        "must_have_met": must_met,
        "good_to_have_total": good_total,
        "good_to_have_coverage_pct": round((good_met / good_total * 100.0), 1) if good_total else 0.0,
        "good_to_have_met": good_met,
        "architecture": arch_json,
        "architecture_error": architecture_error,
    }
    
    return {"summary": summary, "per_intent": results}


# ============================================================================
# PDF Generation
# ============================================================================

def generate_summary_pdf(
    result: Dict[str, Any],
    pdf_output_path: str = "summary_report.pdf",
) -> bool:
    """Generate summary PDF report."""
    try:
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        
        if not (pdf_output_path or "").strip():
            pdf_output_path = "summary_report.pdf"
        
        if not pdf_output_path.lower().endswith(".pdf"):
            pdf_output_path = f"{pdf_output_path}.pdf"
        
        styles = getSampleStyleSheet()
        link_style = styles["Normal"].clone("link_style")
        link_style.textColor = colors.blue
        link_style.underline = True
        
        summary = result.get("summary", {})
        per_intent = result.get("per_intent", [])
        architecture = summary.get("architecture", {}) or {}
        
        nar_id = summary.get("nar_id", "")
        application_name = summary.get("application_name", "")
        release_number = summary.get("release_number", "")
        rtype = summary.get("rtype", "")
        must_have_total = summary.get("must_have_total", 0)
        must_have_met = summary.get("must_have_met", 0)
        must_have_coverage = summary.get("must_have_coverage_pct", 0.0)
        must_have_diff = must_have_total - must_have_met
        
        good_to_have_total = summary.get("good_to_have_total", 0)
        good_to_have_met = summary.get("good_to_have_met", 0)
        good_to_have_diff = good_to_have_total - good_to_have_met
        good_to_have_coverage = summary.get("good_to_have_coverage_pct", 0.0)
        
        architecture_summary = architecture.get("summary", "")
        if not architecture_summary:
            architecture_summary = (
                f"This report is for NAR ID {nar_id}, application {application_name}, "
                f"release {release_number}, and rtype {rtype}."
            )
        
        result_status = "PASS" if must_have_total == must_have_met else "FAIL"
        
        doc = SimpleDocTemplate(
            pdf_output_path,
            pagesize=A4,
            leftMargin=40,
            rightMargin=40,
            topMargin=40,
            bottomMargin=40,
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=22,
            textColor=colors.HexColor("#1a5e80"),
            spaceAfter=18,
            alignment=1,
            fontName="Helvetica-Bold",
        )
        normal_style = styles["BodyText"]
        small_style = ParagraphStyle(
            "SmallStyle",
            parent=styles["BodyText"],
            fontSize=10,
            textColor=colors.HexColor("#333333"),
        )
        
        def section_heading(text):
            tbl = Table(
                [[Paragraph(f"<b>{text}</b>", ParagraphStyle(
                    "SectionHeadingText",
                    parent=styles["BodyText"],
                    fontSize=11,
                    textColor=colors.white,
                    fontName="Helvetica-Bold",
                    alignment=0,
                ))]],
                colwidths=[500],
            )
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#2d7fa3")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            return tbl
        
        story = []
        story.append(Paragraph("SDLC Control Validation Summary", title_style))
        story.append(Spacer(1, 8))
        story.append(section_heading("Document Information"))
        story.append(Spacer(1, 8))
        
        metadata_table_data = [
            [
                Paragraph("<b>NAR ID</b>", normal_style),
                Paragraph(str(nar_id), normal_style),
            ],
            [
                Paragraph("<b>Application</b>", normal_style),
                Paragraph(str(application_name), normal_style),
            ],
            [
                Paragraph("<b>Release</b>", normal_style),
                Paragraph(str(release_number), normal_style),
            ],
            [
                Paragraph("<b>Type</b>", normal_style),
                Paragraph(str(rtype), normal_style),
            ],
        ]
        metadata_table = Table(metadata_table_data, colwidths=[120, 380])
        metadata_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f1f5")),
            ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#f9f9f9")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 12))
        
        story.append(section_heading("Result"))
        story.append(Spacer(1, 6))
        
        reason_text = (
            "All must-have requirements are met."
            if result_status == "PASS"
            else f"{must_have_diff} must-have requirement(s) are not met."
        )
        result_line_style = ParagraphStyle(
            "ResultLineStyle",
            parent=styles["BodyText"],
            fontSize=11,
            textColor=colors.HexColor("#222222"),
            spaceAfter=6,
            leading=14,
        )
        status_color = "#2e7d32" if result_status == "PASS" else "#c62828"
        story.append(
            Paragraph(
                f'Result: <font color="{status_color}"><b>{result_status}</b></font>',
                result_line_style,
            )
        )
        story.append(Paragraph(f"Reason: {reason_text}", result_line_style))
        story.append(Spacer(1, 10))
        
        story.append(section_heading("Must-Have Requirements"))
        story.append(Spacer(1, 8))
        
        metrics_data = [
            [
                Paragraph("<b>Total</b>", small_style),
                Paragraph("<b>Met</b>", small_style),
                Paragraph("<b>Must-Have Not Met</b>", small_style),
                Paragraph("<b>Coverage</b>", small_style),
            ],
            [
                Paragraph(str(must_have_total), small_style),
                Paragraph(str(must_have_met), small_style),
                Paragraph(str(must_have_diff), small_style),
                Paragraph(f"<b>{must_have_coverage:.1f}%</b>", small_style),
            ],
        ]
        metrics_table = Table(metrics_data, colwidths=[80, 80, 160, 80])
        metrics_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9eaf7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f9f9f9")),
            ("GRID", (0, 0), (-1, -1), 0.75, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 12))
        
        story.append(section_heading("Good-to-Have Requirements"))
        story.append(Spacer(1, 8))
        
        good_metrics_data = [
            [
                Paragraph("<b>Total</b>", small_style),
                Paragraph("<b>Met</b>", small_style),
                Paragraph("<b>Good-to-Have Not Met</b>", small_style),
                Paragraph("<b>Coverage</b>", small_style),
            ],
            [
                Paragraph(str(good_to_have_total), small_style),
                Paragraph(str(good_to_have_met), small_style),
                Paragraph(str(good_to_have_diff), small_style),
                Paragraph(f"<b>{good_to_have_coverage:.1f}%</b>", small_style),
            ],
        ]
        good_metrics_table = Table(good_metrics_data, colwidths=[80, 80, 160, 80])
        good_metrics_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9eaf7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f9f9f9")),
            ("GRID", (0, 0), (-1, -1), 0.75, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(good_metrics_table)
        story.append(Spacer(1, 12))
        
        story.append(section_heading("Justification"))
        story.append(Spacer(1, 8))
        
        table_data = [[Paragraph("<b>Name</b>", normal_style), Paragraph("<b>Justification</b>", normal_style)]]
        found_rows = 0
        
        for item in per_intent:
            if item.get("musthave") and item.get("status") != "PRESENT":
                name = item.get("tag", "") or item.get("intent", "") or "N/A"
                status_val = item.get("status", "")
                justification = item.get("justification", "") or "No justification available."
                justification_text = f"[{status_val}] {justification}" if status_val else justification
                table_data.append([
                    Paragraph(str(name), normal_style),
                    Paragraph(str(justification_text), normal_style),
                ])
                found_rows += 1
        
        if found_rows == 0:
            table_data.append([
                Paragraph("No Gap", normal_style),
                Paragraph(
                    "All must-have requirements are PRESENT, so there is no justification gap.",
                    normal_style,
                ),
            ])
        
        justification_table = Table(table_data, colwidths=[160, 330])
        justification_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9eaf7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.75, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(justification_table)
        story.append(Spacer(1, 16))
        
        story.append(section_heading("Architecture Overview"))
        story.append(Spacer(1, 8))
        story.append(Paragraph(architecture_summary, normal_style))
        story.append(Spacer(1, 12))
        
        footer_text = f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        footer_style = ParagraphStyle(
            "FooterStyle",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.grey,
            alignment=2,
        )
        story.append(Paragraph(footer_text, footer_style))
        
        doc.build(story)
        logging.info(f"Enhanced PDF Summary Report generated: {pdf_output_path}")
        return True
    
    except Exception as e:
        logging.error(f"Failed to generate PDF summary report: {e}")
        return False


# ============================================================================
# CLI
# ============================================================================

def main():
    """Command-line interface."""
    ap = argparse.ArgumentParser(
        description="Quality-aware Validation using Template JSON + Template PDF Evidence (text+diagrams)"
    )
    ap.add_argument("--template-json", required=True, help="Path to Template JSON (e.g., Template_V5.json)")
    ap.add_argument("--template-pdf", required=True, help="Path to Template PDF/DOCX with instructions")
    ap.add_argument("--nar-id", required=True)
    ap.add_argument("--application-name", required=True)
    ap.add_argument("--release-number", required=True)
    ap.add_argument("--rtype", required=True, help="indev | new | migration | rehost | normal")
    ap.add_argument("--top-k", type=int, default=6, help="Top-K evidence text chunks per tag")
    ap.add_argument("--scope-doc-id", default="", help="Optional: restrict validation to a specific evidence doc_id")
    ap.add_argument("--scope-doc-hash", default="", help="Deprecated alias for --scope-doc-id")
    ap.add_argument("--enable-mm", action="store_true", help="Enable multimodal lane (diagram checks)")
    ap.add_argument("--out", default="validation_quality_result.json", help="Output JSON path")
    ap.add_argument("--pdf-out", default="summary_report.pdf", help="Output PDF path")
    
    args = ap.parse_args()
    
    rtype = resolve_rtype(args.rtype)
    logging.info(f"[CFG] Using rtype='{rtype}', TOP-K={args.top_k}, MM={args.enable_mm}")
    
    # Fail fast on DB connection
    get_engine()
    validate_runtime_contract(MM_DIM)
    
    scope_doc_id = args.scope_doc_id.strip() or args.scope_doc_hash.strip() or None
    if args.scope_doc_hash.strip() and not args.scope_doc_id.strip():
        logging.warning("--scope-doc-hash is deprecated; use --scope-doc-id instead.")

    if scope_doc_id:
        logging.info(f"Using explicit doc_id scope: {scope_doc_id}")
    else:
        logging.info(
            "No doc_id scope provided; validating across matching evidence for the selected NAR/release/rtype."
        )
    
    result = run_validation(
        template_json_path=args.template_json,
        template_pdf_path=args.template_pdf,
        nar_id=args.nar_id,
        application_name=args.application_name,
        release_number=args.release_number,
        rtype=rtype,
        top_k=args.top_k,
        scope_doc_id=scope_doc_id,
        enable_mm=args.enable_mm,
    )
    
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logging.info(f"Done. JSON: {args.out}")
    
    pdf_output_path = args.pdf_out if args.pdf_out.lower().endswith(".pdf") else f"{args.pdf_out}.pdf"
    pdf_ok = generate_summary_pdf(result, pdf_output_path)
    
    if pdf_ok:
        logging.info(f"Done. PDF: {pdf_output_path}")
    else:
        logging.error("PDF generation failed")


if __name__ == "__main__":
    main()
