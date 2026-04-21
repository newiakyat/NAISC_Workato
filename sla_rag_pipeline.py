#!/usr/bin/env python3
"""
sla_rag_pipeline.py
═══════════════════════════════════════════════════════════════════════════════
NovaSuite SLA RAG Pipeline — Workato Genie Agent Skills
───────────────────────────────────────────────────────────────────────────────
SKILL 1 (ingest):
  PDF → Markdown (with headers) → Structural Chunking → Embeddings → ChromaDB
  → Packages PDF + chroma_db.zip for Dropbox upload

SKILL 2 (query):
  Incident details → ChromaDB retrieval → SLA maths → Slack-ready assessment

USAGE:
  # Ingest
  python sla_rag_pipeline.py ingest \
      --pdf /path/to/sla.pdf \
      --chroma-dir ./chroma_db \
      --output-dir ./output

  # Query
  python sla_rag_pipeline.py query \
      --tier enterprise \
      --incident-start 2026-04-21T08:00:00Z \
      --current-time 2026-04-21T08:17:00Z \
      --service "NovaSuite Platform" \
      --chroma-source ./output/chroma_db.zip \
      --prior-downtime-min 10.5 \
      --output-format json

OUTPUT: JSON printed to stdout (Workato parses this via datapill)
LOGS:   Written to stderr (visible in Workato job logs)

REQUIREMENTS (pip install):
  pymupdf4llm>=0.0.12
  PyMuPDF>=1.24.0
  langchain-text-splitters>=0.2.0
  sentence-transformers>=2.7.0
  chromadb>=0.5.0
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import json
import shutil
import zipfile
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

# ─── Third-party imports ──────────────────────────────────────────────────────
try:
    import pymupdf4llm                          # Best-in-class PDF → Markdown
    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    PYMUPDF4LLM_AVAILABLE = False

try:
    import fitz                                 # PyMuPDF fallback
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from sentence_transformers import SentenceTransformer
import chromadb

# ─── Pipeline configuration ───────────────────────────────────────────────────
EMBED_MODEL      = "all-MiniLM-L6-v2"      # 384-dim, fast, good quality, runs locally
COLLECTION_NAME  = "sla_documents"
DEFAULT_CHROMA   = "./chroma_db"
DEFAULT_OUTPUT   = "./output"
CHUNK_SIZE       = 800                      # characters
CHUNK_OVERLAP    = 100
TOP_K_RETRIEVAL  = 6

HEADERS_TO_SPLIT = [
    ("#",   "h1"),
    ("##",  "h2"),
    ("###", "h3"),
]

# ─── SLA reference data (mirrors the NovaSuite SLA document) ─────────────────
SLA_TIERS = {
    "starter":      {"uptime_pct": 99.5,   "max_downtime_min": 219.0,  "credits": False},
    "professional": {"uptime_pct": 99.9,   "max_downtime_min": 43.8,   "credits": True},
    "enterprise":   {"uptime_pct": 99.95,  "max_downtime_min": 21.9,   "credits": True},
}

# Section 5.1 credit bands (Professional & Enterprise only)
CREDIT_BANDS = [
    {"min": 99.0,  "max": 99.89, "pct": 10},
    {"min": 95.0,  "max": 98.99, "pct": 25},
    {"min":  0.0,  "max": 94.99, "pct": 50},
]

# Section 4.1 support response times
SUPPORT_SLA = {
    "starter": {
        "P1": "4 hours",   "P2": "8 hours",
        "P3": "2 business days", "P4": "5 business days",
        "channels": "Email / Ticket Portal",
        "hours": "Business hours (9–5 ET)",
    },
    "professional": {
        "P1": "1 hour",    "P2": "4 hours",
        "P3": "1 business day",  "P4": "3 business days",
        "channels": "Email / Ticket Portal, In-App Live Chat",
        "hours": "Extended (7am–11pm ET)",
    },
    "enterprise": {
        "P1": "30 minutes", "P2": "2 hours",
        "P3": "4 hours",     "P4": "1 business day",
        "channels": "Email / Ticket, Live Chat, Phone/Video, Named CSM",
        "hours": "24 × 7 × 365",
    },
}

# Data recovery objectives (Section 6.2)
RECOVERY_OBJECTIVES = {
    "professional": {"rto": "4 hours", "rpo": "24 hours"},
    "enterprise":   {"rto": "1 hour",  "rpo": "1 hour"},
}


# ═══════════════════════════════════════════════════════════════════════════════
# SKILL 1 — INGEST
# ═══════════════════════════════════════════════════════════════════════════════

def pdf_to_markdown(pdf_path: str) -> str:
    """
    Convert a PDF to structured Markdown, preserving section hierarchy.

    Strategy:
    1. Try pymupdf4llm (best fidelity, handles tables/columns)
    2. Fall back to PyMuPDF with heuristic header detection via font size/weight
    """
    if PYMUPDF4LLM_AVAILABLE:
        log(f"Converting PDF with pymupdf4llm: {pdf_path}")
        md = pymupdf4llm.to_markdown(pdf_path)
        return md

    if FITZ_AVAILABLE:
        log("pymupdf4llm not found — falling back to PyMuPDF heuristic extraction")
        doc = fitz.open(pdf_path)
        lines = []

        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                        size  = span["size"]
                        bold  = bool(span["flags"] & 16)

                        # Heuristic: map font size → Markdown heading level
                        if size >= 16 and bold:
                            lines.append(f"# {text}")
                        elif size >= 13 and bold:
                            lines.append(f"## {text}")
                        elif size >= 11 and bold:
                            lines.append(f"### {text}")
                        else:
                            lines.append(text)
        doc.close()
        return "\n\n".join(lines)

    raise RuntimeError(
        "No PDF library available. "
        "Install: pip install pymupdf4llm   (or: pip install PyMuPDF)"
    )


def structural_chunk(markdown_text: str) -> list:
    """
    Two-pass structural chunking:

    Pass 1 — MarkdownHeaderTextSplitter
        Splits at H1 / H2 / H3 boundaries, carrying header metadata on each chunk.
        This gives us semantically coherent SLA sections (e.g. "Service Credits",
        "Uptime Targets") as individual retrieval units.

    Pass 2 — RecursiveCharacterTextSplitter
        Any section that still exceeds CHUNK_SIZE characters is further split
        with overlap to preserve sentence continuity.
    """
    log("Pass 1: MarkdownHeaderTextSplitter")
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,            # keep headers in chunk text for context
        return_each_line=False,
    )
    header_chunks = header_splitter.split_text(markdown_text)
    log(f"  → {len(header_chunks)} header-based chunks")

    log("Pass 2: RecursiveCharacterTextSplitter (max {CHUNK_SIZE} chars)")
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    final_chunks = []
    for chunk in header_chunks:
        sub = char_splitter.split_documents([chunk])
        final_chunks.extend(sub)

    log(f"  → {len(final_chunks)} final chunks after size enforcement")
    return final_chunks


def embed_and_store(chunks: list, chroma_dir: str) -> int:
    """
    Generate sentence-transformer embeddings (cosine space) and
    persist to a ChromaDB collection.

    Uses batched upserts to avoid memory spikes on large SLA documents.
    """
    log(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    log(f"Initialising ChromaDB at: {chroma_dir}")
    client = chromadb.PersistentClient(path=chroma_dir)

    # Fresh collection for this SLA document
    try:
        client.delete_collection(name=COLLECTION_NAME)
        log("  Existing collection cleared")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Build lists
    texts, metadatas, ids, embeddings = [], [], [], []
    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        # Sanitise metadata values to strings (ChromaDB requirement)
        meta = {k: str(v) for k, v in chunk.metadata.items()}
        emb  = model.encode(text, normalize_embeddings=True).tolist()

        texts.append(text)
        metadatas.append(meta)
        ids.append(f"chunk_{i:05d}")
        embeddings.append(emb)

    # Batch upsert in groups of 100
    BATCH = 100
    for start in range(0, len(texts), BATCH):
        end = start + BATCH
        collection.add(
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        log(f"  Stored batch {start}–{min(end, len(texts))}")

    log(f"ChromaDB ready: {len(texts)} vectors in collection '{COLLECTION_NAME}'")
    return len(texts)


def package_for_dropbox(pdf_path: str, chroma_dir: str, output_dir: str) -> dict:
    """
    Prepare two artefacts for Workato to upload to Dropbox:
      1. The original PDF (renamed with timestamp)
      2. A ZIP of the ChromaDB persistence directory

    Workato's Dropbox connector reads local file paths, so we stage
    everything under output_dir before the recipe uploads them.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Copy PDF with timestamp suffix so multiple SLAs don't clobber each other
    stem = Path(pdf_path).stem
    pdf_dest = os.path.join(output_dir, f"{stem}_{ts}.pdf")
    shutil.copy2(pdf_path, pdf_dest)
    log(f"PDF staged → {pdf_dest}")

    # Zip the ChromaDB directory
    zip_dest = os.path.join(output_dir, f"chroma_db_{ts}.zip")
    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(chroma_dir):
            for file in files:
                fp      = os.path.join(root, file)
                arcname = os.path.relpath(fp, os.path.dirname(chroma_dir))
                zf.write(fp, arcname)
    log(f"ChromaDB zipped → {zip_dest}")

    return {
        "pdf_output_path":   pdf_dest,
        "chroma_zip_path":   zip_dest,
        "pdf_filename":      os.path.basename(pdf_dest),
        "chroma_zip_filename": os.path.basename(zip_dest),
    }


def run_ingest(pdf_path: str, chroma_dir: str, output_dir: str) -> dict:
    """
    Full Skill 1 pipeline.
    Returns a JSON-serialisable dict that Workato maps to datapills.
    """
    log("━━━ SKILL 1: SLA INGESTION STARTED ━━━")

    markdown      = pdf_to_markdown(pdf_path)
    log(f"Markdown extracted: {len(markdown):,} characters")

    chunks        = structural_chunk(markdown)
    n_stored      = embed_and_store(chunks, chroma_dir)
    paths         = package_for_dropbox(pdf_path, chroma_dir, output_dir)

    log("━━━ INGESTION COMPLETE ━━━")
    return {
        "status":              "success",
        "mode":                "ingest",
        "source_pdf":          pdf_path,
        "chunks_created":      n_stored,
        "chroma_db_dir":       chroma_dir,
        **paths,
        # Workato datapills: use pdf_output_path / chroma_zip_path for Dropbox upload
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SKILL 2 — QUERY + SLA ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════

def unzip_chroma(zip_path: str, extract_to: str = "./chroma_db_live") -> str:
    """Extract a Dropbox-downloaded ChromaDB zip to a local directory."""
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    log(f"ChromaDB extracted to: {extract_to}")
    return extract_to


def retrieve_sla_context(query: str, chroma_dir: str) -> list[dict]:
    """
    Embed the query and perform cosine-similarity retrieval against the
    ChromaDB collection, returning the top-K most relevant SLA chunks.
    """
    model  = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=chroma_dir)
    col    = client.get_collection(name=COLLECTION_NAME)

    q_emb  = model.encode(query, normalize_embeddings=True).tolist()
    result = col.query(query_embeddings=[q_emb], n_results=TOP_K_RETRIEVAL)

    chunks = []
    for i, text in enumerate(result["documents"][0]):
        chunks.append({
            "text":     text,
            "metadata": result["metadatas"][0][i],
            "score":    round(1 - result["distances"][0][i], 4),  # cosine similarity
        })
    log(f"Retrieved {len(chunks)} SLA context chunks")
    return chunks


def _minutes_in_month(dt: datetime) -> float:
    """Return the total number of minutes in the calendar month of dt."""
    year, month = dt.year, dt.month
    if month == 12:
        next_month = datetime(year + 1, 1, 1, tzinfo=dt.tzinfo)
    else:
        next_month = datetime(year, month + 1, 1, tzinfo=dt.tzinfo)
    this_month = datetime(year, month, 1, tzinfo=dt.tzinfo)
    return (next_month - this_month).total_seconds() / 60


def calculate_sla_impact(
    tier: str,
    incident_start_iso: str,
    current_time_iso: str,
    prior_downtime_min: float,
) -> dict:
    """
    Deterministic SLA impact calculation based on Section 3 and Section 5.

    Returns all values needed to compose the Slack assessment message.
    """
    tier_cfg = SLA_TIERS.get(tier.lower())
    if not tier_cfg:
        return {"error": f"Unknown tier '{tier}'. Valid values: starter, professional, enterprise"}

    # ── Time arithmetic ──────────────────────────────────────────────────────
    t_start   = datetime.fromisoformat(incident_start_iso.replace("Z", "+00:00"))
    t_now     = datetime.fromisoformat(current_time_iso.replace("Z", "+00:00"))
    incident_duration_min  = (t_now - t_start).total_seconds() / 60

    total_downtime_min     = prior_downtime_min + incident_duration_min
    total_minutes_in_month = _minutes_in_month(t_now)
    achieved_uptime_pct    = (
        (total_minutes_in_month - total_downtime_min) / total_minutes_in_month
    ) * 100

    # ── Breach status ────────────────────────────────────────────────────────
    max_dt_min         = tier_cfg["max_downtime_min"]
    already_breached   = total_downtime_min > max_dt_min
    minutes_until_breach = max(0.0, max_dt_min - total_downtime_min)
    overage_min        = max(0.0, total_downtime_min - max_dt_min)

    # ── Credit eligibility (Sections 5.1, 5.3) ──────────────────────────────
    credit_pct       = 0
    credit_triggered = False
    if already_breached and tier_cfg["credits"]:
        for band in CREDIT_BANDS:
            if band["min"] <= achieved_uptime_pct <= band["max"]:
                credit_pct = band["pct"]
                credit_triggered = True
                break

    # ── Risk level ───────────────────────────────────────────────────────────
    if already_breached:
        risk = "BREACHED"
    elif minutes_until_breach <= 2:
        risk = "CRITICAL"
    elif minutes_until_breach <= max_dt_min * 0.25:
        risk = "HIGH"
    elif minutes_until_breach <= max_dt_min * 0.50:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # ── RCA obligation (Section 8) ───────────────────────────────────────────
    rca_due_by = None
    if incident_duration_min >= 1:   # any confirmed P1
        rca_dt   = t_now + timedelta(days=5)
        rca_due_by = rca_dt.strftime("%Y-%m-%d")

    return {
        "tier":                        tier.lower(),
        "sla_uptime_target_pct":       tier_cfg["uptime_pct"],
        "max_allowable_downtime_min":  max_dt_min,
        "incident_duration_min":       round(incident_duration_min, 2),
        "prior_downtime_this_month_min": round(prior_downtime_min, 2),
        "total_downtime_this_month_min": round(total_downtime_min, 2),
        "achieved_uptime_pct":         round(achieved_uptime_pct, 5),
        "remaining_budget_min":        round(minutes_until_breach, 2),
        "overage_min":                 round(overage_min, 2),
        "sla_breached":                already_breached,
        "risk_level":                  risk,
        "credit_eligible":             credit_triggered,
        "credit_pct":                  credit_pct,
        "credit_claim_window":         "30 calendar days after month end",
        "credit_claim_url":            "https://support.novasuite.io/credits",
        "starter_credit_note":         "Starter tier is not eligible for service credits (Section 5.1)" if tier.lower() == "starter" else None,
        "support_sla":                 SUPPORT_SLA.get(tier.lower(), {}),
        "recovery_objectives":         RECOVERY_OBJECTIVES.get(tier.lower(), {}),
        "rca_required":                tier.lower() != "starter" and incident_duration_min >= 1,
        "rca_due_by":                  rca_due_by,
        "status_page":                 "https://status.novasuite.io",
        "credit_max_cap":              "50% of monthly subscription fee (Section 5.3)",
    }


def _risk_emoji(risk: str) -> str:
    return {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴", "BREACHED": "💥"}.get(risk, "⚪")


def format_slack_message(assessment: dict, sla_chunks: list[dict], service_name: str) -> str:
    """
    Compose a structured, emoji-rich Slack message from the assessment dict.
    Designed to be pasted directly into a Workato 'Post message' Slack action.
    """
    a     = assessment
    risk  = a.get("risk_level", "UNKNOWN")
    emoji = _risk_emoji(risk)

    lines = [
        f"*🚨 SLA Impact Assessment — {service_name}*",
        f"*Tier:* `{a['tier'].title()}`   *Risk:* {emoji} `{risk}`",
        "─" * 48,
        "",
        "*📊 Uptime Status*",
        f"> SLA Target: `{a['sla_uptime_target_pct']}%` uptime",
        f"> Max Allowable Downtime: `{a['max_allowable_downtime_min']} min / month`",
        f"> Achieved Uptime (this month): `{a['achieved_uptime_pct']}%`",
        "",
        "*⏱ Incident Timeline*",
        f"> Current Incident Duration: `{a['incident_duration_min']} min`",
        f"> Prior Downtime This Month: `{a['prior_downtime_this_month_min']} min`",
        f"> *Total Downtime This Month: `{a['total_downtime_this_month_min']} min`*",
        "",
    ]

    # ── Breach / Budget block ─────────────────────────────────────────────
    if a["sla_breached"]:
        lines += [
            f"*⚠️  SLA BREACHED — Exceeded by `{a['overage_min']} min`*",
        ]
        if a["credit_eligible"]:
            lines += [
                "",
                "*💳 Service Credit Triggered*",
                f"> Credit: `{a['credit_pct']}%` of monthly subscription fee",
                f"> Cap: {a['credit_max_cap']}",
                f"> Claim window: {a['credit_claim_window']}",
                f"> Submit at: {a['credit_claim_url']}",
            ]
        elif a.get("starter_credit_note"):
            lines.append(f"> ℹ️  {a['starter_credit_note']}")
    else:
        lines += [
            f"*⏳ Time Remaining Before SLA Breach: `{a['remaining_budget_min']} min`*",
        ]

    # ── Support SLA ───────────────────────────────────────────────────────
    sup = a.get("support_sla", {})
    if sup:
        lines += [
            "",
            "*📞 Support Response SLA*",
            f"> P1 Critical: `{sup.get('P1', 'N/A')}`  |  P2 High: `{sup.get('P2', 'N/A')}`",
            f"> P3 Medium: `{sup.get('P3', 'N/A')}`  |  P4 Low: `{sup.get('P4', 'N/A')}`",
            f"> Channels: {sup.get('channels', 'N/A')}",
            f"> Coverage: {sup.get('hours', 'N/A')}",
        ]

    # ── Recovery objectives ───────────────────────────────────────────────
    rco = a.get("recovery_objectives", {})
    if rco:
        lines += [
            "",
            "*💾 Data Recovery Objectives*",
            f"> RTO: `{rco.get('rto', 'N/A')}`   RPO: `{rco.get('rpo', 'N/A')}`",
        ]

    # ── RCA obligation ────────────────────────────────────────────────────
    if a.get("rca_required"):
        lines += [
            "",
            f"*📝 Root Cause Analysis due by: `{a['rca_due_by']}`* (Section 8)",
        ]

    # ── RAG context citations ─────────────────────────────────────────────
    if sla_chunks:
        lines += ["", "*📄 Relevant SLA Clauses (retrieved from document)*"]
        for i, chunk in enumerate(sla_chunks[:3], 1):
            section = (
                chunk["metadata"].get("h2")
                or chunk["metadata"].get("h1")
                or "SLA Document"
            )
            preview = chunk["text"][:220].replace("\n", " ").strip()
            score   = chunk.get("score", 0)
            lines.append(f"  *{i}. [{section}]* (relevance: {score:.2f})")
            lines.append(f"     _{preview}…_")

    lines += ["", f"_Status page: {a['status_page']}_"]
    return "\n".join(lines)


def run_query(
    tier: str,
    incident_start_iso: str,
    current_time_iso: str,
    service_name: str,
    chroma_source: str,
    prior_downtime_min: float,
    output_format: str,
) -> dict:
    """
    Full Skill 2 pipeline.
    Returns a JSON-serialisable dict that Workato maps to datapills.
    """
    log("━━━ SKILL 2: SLA QUERY STARTED ━━━")

    # ── Prepare ChromaDB ──────────────────────────────────────────────────────
    if chroma_source.endswith(".zip"):
        chroma_dir = unzip_chroma(chroma_source)
    else:
        chroma_dir = chroma_source

    # ── Semantic retrieval query ──────────────────────────────────────────────
    # Intentionally broad: covers uptime, credits, incident response, support
    retrieval_query = (
        f"Monthly uptime SLA commitment, downtime allowance, service credit eligibility, "
        f"incident response obligations, support response times for {tier} subscription tier. "
        "SLA breach remedies, scheduled maintenance, excused outages."
    )
    sla_chunks = retrieve_sla_context(retrieval_query, chroma_dir)

    # ── SLA impact calculation ────────────────────────────────────────────────
    assessment  = calculate_sla_impact(
        tier=tier,
        incident_start_iso=incident_start_iso,
        current_time_iso=current_time_iso,
        prior_downtime_min=prior_downtime_min,
    )

    slack_msg = format_slack_message(assessment, sla_chunks, service_name)

    log("━━━ QUERY COMPLETE ━━━")

    return {
        "status":         "success",
        "mode":           "query",
        "service_name":   service_name,
        "assessment":     assessment,
        "slack_message":  slack_msg,
        "sla_chunks":     sla_chunks,   # raw chunks for Workato datapill / logging
    }


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def log(msg: str):
    """Write to stderr so Workato job logs capture it without polluting stdout JSON."""
    print(f"[SLA-RAG] {msg}", file=sys.stderr, flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT (called by Workato "Run Python script" action)
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sla_rag_pipeline",
        description="SLA RAG Pipeline — NovaSuite / Workato Agent Skills",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── Ingest ────────────────────────────────────────────────────────────────
    p_ingest = sub.add_parser("ingest", help="Ingest SLA PDF into ChromaDB")
    p_ingest.add_argument("--pdf",        required=True, help="Path to the SLA PDF file")
    p_ingest.add_argument("--chroma-dir", default=DEFAULT_CHROMA, help="ChromaDB persistence directory")
    p_ingest.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Staging directory for Dropbox upload")

    # ── Query ─────────────────────────────────────────────────────────────────
    p_query = sub.add_parser("query", help="Assess SLA impact for an active incident")
    p_query.add_argument("--tier",         required=True, choices=["starter", "professional", "enterprise"])
    p_query.add_argument("--incident-start", required=True,
                         help="Incident start time (ISO 8601, e.g. 2026-04-21T08:00:00Z)")
    p_query.add_argument("--current-time",   required=True,
                         help="Current timestamp (ISO 8601)")
    p_query.add_argument("--service",        default="NovaSuite Platform",
                         help="Affected service / product name")
    p_query.add_argument("--chroma-source",  required=True,
                         help="Path to ChromaDB directory OR chroma_db.zip downloaded from Dropbox")
    p_query.add_argument("--prior-downtime-min", type=float, default=0.0,
                         help="Minutes of recorded downtime this month BEFORE the current incident")
    p_query.add_argument("--output-format",  choices=["json", "slack"], default="json",
                         help="'json' = full JSON (for Workato datapills); 'slack' = only the Slack message")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.mode == "ingest":
        result = run_ingest(
            pdf_path=args.pdf,
            chroma_dir=args.chroma_dir,
            output_dir=args.output_dir,
        )
    else:  # query
        result = run_query(
            tier=args.tier,
            incident_start_iso=args.incident_start,
            current_time_iso=args.current_time,
            service_name=args.service,
            chroma_source=args.chroma_source,
            prior_downtime_min=args.prior_downtime_min,
            output_format=args.output_format,
        )

        if args.output_format == "slack":
            # Workato can feed this directly to "Post message" action
            print(result["slack_message"])
            return

    # Default: full JSON to stdout — Workato parses via "Parse JSON" action
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
