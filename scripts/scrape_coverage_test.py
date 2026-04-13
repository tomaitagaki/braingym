"""
Test open-access coverage for BrainBiz corpus.

Three-layer strategy:
  1. Keyword filters  — OpenAlex auto-assigned keywords (zero false positives)
  2. Keyword + search — narrow within domain (e.g., neuromarketing + "fMRI")
  3. Semantic search   — AI embedding similarity (highest precision, max 50/query)

Usage: python3 scripts/scrape_coverage_test.py
"""

import json
import time
from pathlib import Path
from collections import Counter

from pyalex import Works, config

config.email = "tom@implicit.co"

OUTPUT_DIR = Path("cache/brainbiz_corpus")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── OpenAlex keyword IDs (auto-assigned to papers, high precision) ──────────
KEYWORD_IDS = [
    "https://openalex.org/keywords/neuromarketing",
    "https://openalex.org/keywords/consumer-neuroscience",
    "https://openalex.org/keywords/neuroeconomics",
]

# ── Narrow search terms to combine WITH keyword filters ─────────────────────
# These run as: filter(keyword=X) + .search(term)
KEYWORD_PLUS_SEARCH = [
    # Brain imaging specific
    ("https://openalex.org/keywords/neuromarketing", "fMRI"),
    ("https://openalex.org/keywords/neuromarketing", "EEG"),
    ("https://openalex.org/keywords/neuroeconomics", "fMRI prediction"),
    ("https://openalex.org/keywords/neuroeconomics", "reward anticipation"),
    # Engagement / attention
    ("https://openalex.org/keywords/neuroimaging", "advertising effectiveness"),
    ("https://openalex.org/keywords/neuroimaging", "consumer engagement"),
    ("https://openalex.org/keywords/neuroimaging", "video engagement attention"),
    # Prediction
    ("https://openalex.org/keywords/functional-magnetic-resonance-imaging", "neuroforecasting"),
    ("https://openalex.org/keywords/functional-magnetic-resonance-imaging", "predict market"),
    ("https://openalex.org/keywords/functional-magnetic-resonance-imaging", "predict viral sharing"),
    ("https://openalex.org/keywords/nucleus-accumbens", "consumer choice purchasing"),
]

# ── Semantic search queries (highest precision, max 50 results each) ────────
SEMANTIC_QUERIES = [
    "neuroforecasting brain imaging predicts aggregate market outcomes",
    "fMRI neural activity predicts consumer purchasing decisions",
    "brain responses predict box office movie success",
    "nucleus accumbens ventral striatum predict purchasing behavior",
    "EEG frontal asymmetry predicts advertising effectiveness",
    "neural prediction viral content sharing social media",
    "inter-subject correlation engagement naturalistic movie viewing",
    "brain encoding model predicts video attention retention",
    "neuroforecasting crowdfunding kickstarter campaign success",
    "fMRI memorability prediction subsequent memory effect advertising",
    "brain activity content engagement prediction population behavior",
    "neural correlates brand preference consumer decision making",
]


def extract_paper(work: dict, source_label: str) -> dict:
    """Extract consistent paper record from an OpenAlex work."""
    # Reconstruct abstract from inverted index if available
    abstract = work.get("abstract")
    if not abstract:
        inv = work.get("abstract_inverted_index")
        if inv:
            positions = {}
            for word, idxs in inv.items():
                for idx in idxs:
                    positions[idx] = word
            if positions:
                abstract = " ".join(positions[i] for i in sorted(positions.keys()))

    keywords = []
    for kw in (work.get("keywords") or []):
        if isinstance(kw, dict):
            keywords.append(kw.get("display_name", ""))
        elif isinstance(kw, str):
            keywords.append(kw)

    src = (work.get("primary_location", {}) or {}).get("source", {}) or {}

    return {
        "id": work.get("id", ""),
        "doi": work.get("doi", ""),
        "title": work.get("title", ""),
        "publication_year": work.get("publication_year"),
        "cited_by_count": work.get("cited_by_count", 0),
        "is_oa": work.get("open_access", {}).get("is_oa", False),
        "oa_status": work.get("open_access", {}).get("oa_status", "closed"),
        "oa_url": work.get("open_access", {}).get("oa_url"),
        "source_name": src.get("display_name", ""),
        "source_issn": src.get("issn_l", ""),
        "source_type": src.get("type", ""),
        "abstract": abstract,
        "keywords": keywords,
        "query_source": source_label,
    }


def collect(all_papers: dict, works: list, source_label: str) -> int:
    """Add works to all_papers dict (keyed by OpenAlex ID). Returns count of new."""
    new = 0
    for w in works:
        oa_id = w.get("id", "")
        if oa_id and oa_id not in all_papers:
            all_papers[oa_id] = extract_paper(w, source_label)
            new += 1
    return new


def main():
    all_papers = {}  # keyed by OpenAlex ID

    # ── Layer 1: Keyword filters (broad, high recall) ──────────────────────
    print("=" * 60)
    print("LAYER 1: Keyword filters")
    print("=" * 60)

    for kw_id in KEYWORD_IDS:
        kw_name = kw_id.split("/")[-1]
        print(f"\n  Keyword: {kw_name}")
        try:
            results = list(
                Works()
                .filter(keywords={"id": kw_id}, type="article")
                .sort(cited_by_count="desc")
                .paginate(per_page=200, n_max=500)
            )
            new = collect(all_papers, results, f"keyword:{kw_name}")
            print(f"    -> {len(results)} results, {new} new (total: {len(all_papers)})")
        except Exception as e:
            print(f"    -> Error: {e}")
        time.sleep(0.3)

    # ── Layer 2: Keyword + search (targeted slices) ────────────────────────
    print(f"\n{'=' * 60}")
    print("LAYER 2: Keyword + search filters")
    print("=" * 60)

    for kw_id, search_term in KEYWORD_PLUS_SEARCH:
        kw_name = kw_id.split("/")[-1]
        label = f"{kw_name}+{search_term}"
        print(f"\n  {label}")
        try:
            results = list(
                Works()
                .filter(keywords={"id": kw_id}, type="article")
                .search(search_term)
                .sort(cited_by_count="desc")
                .get(per_page=50)
            )
            new = collect(all_papers, results, f"kw+search:{label}")
            print(f"    -> {len(results)} results, {new} new (total: {len(all_papers)})")
        except Exception as e:
            print(f"    -> Error: {e}")
        time.sleep(0.3)

    # ── Layer 3: Semantic search (highest precision) ───────────────────────
    print(f"\n{'=' * 60}")
    print("LAYER 3: Semantic search")
    print("=" * 60)

    for query in SEMANTIC_QUERIES:
        print(f"\n  \"{query[:60]}...\"")
        try:
            results = list(Works().similar(query).get(per_page=50))
            new = collect(all_papers, results, f"semantic:{query[:40]}")
            print(f"    -> {len(results)} results, {new} new (total: {len(all_papers)})")
        except Exception as e:
            print(f"    -> Error: {e}")
        time.sleep(1.5)  # semantic search rate limit: 1 req/sec

    # ── Analysis ───────────────────────────────────────────────────────────
    papers = list(all_papers.values())
    print(f"\n{'=' * 60}")
    print(f"TOTAL UNIQUE PAPERS: {len(papers)}")
    print(f"{'=' * 60}\n")

    # OA status
    oa_statuses = Counter(p["oa_status"] for p in papers)
    print("Open Access Status:")
    for status, count in oa_statuses.most_common():
        pct = count / len(papers) * 100
        print(f"  {status:15s} {count:5d}  ({pct:.1f}%)")

    oa_count = sum(1 for p in papers if p["is_oa"])
    closed_count = len(papers) - oa_count
    print(f"\n  OPEN ACCESS:     {oa_count}/{len(papers)} ({oa_count/len(papers)*100:.1f}%)")
    print(f"  CLOSED:          {closed_count}/{len(papers)} ({closed_count/len(papers)*100:.1f}%)")

    # Abstracts
    has_abstract = sum(1 for p in papers if p["abstract"])
    print(f"\n  HAS ABSTRACT:    {has_abstract}/{len(papers)} ({has_abstract/len(papers)*100:.1f}%)")

    has_url = sum(1 for p in papers if p["oa_url"])
    print(f"  HAS OA URL:      {has_url}/{len(papers)} ({has_url/len(papers)*100:.1f}%)")

    # Year distribution
    print("\nYear distribution:")
    years = Counter(p["publication_year"] for p in papers if p["publication_year"] and p["publication_year"] >= 2005)
    for year in sorted(years.keys()):
        bar = "#" * (years[year] // 3)
        print(f"  {year}: {years[year]:4d} {bar}")

    # Top journals
    print("\nTop 20 journals:")
    journals = Counter(p["source_name"] for p in papers if p["source_name"])
    for journal, count in journals.most_common(20):
        oa_in_j = sum(1 for p in papers if p["source_name"] == journal and p["is_oa"])
        print(f"  {count:4d} ({oa_in_j:3d} OA) | {journal}")

    # Top cited
    print("\nTop 30 most-cited papers:")
    for p in sorted(papers, key=lambda x: x["cited_by_count"], reverse=True)[:30]:
        oa_tag = "OA" if p["is_oa"] else "  "
        year = p["publication_year"] or "?"
        print(f"  [{oa_tag}] {p['cited_by_count']:6d} cites | {year} | {(p['title'] or 'No title')[:85]}")

    # Keywords frequency (what OpenAlex thinks these papers are about)
    print("\nTop 30 keywords across corpus:")
    kw_counter = Counter()
    for p in papers:
        for kw in p.get("keywords", []):
            if kw:
                kw_counter[kw] += 1
    for kw, count in kw_counter.most_common(30):
        print(f"  {count:4d} | {kw}")

    # Query source breakdown
    print("\nPapers by source layer:")
    source_counter = Counter()
    for p in papers:
        layer = p["query_source"].split(":")[0]
        source_counter[layer] += 1
    for src, count in source_counter.most_common():
        print(f"  {src:15s} {count:5d}")

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / "coverage_test_v2.json"
    with open(out_path, "w") as f:
        json.dump(papers, f, indent=2, default=str)
    print(f"\nFull results saved to {out_path}")

    oa_papers = [p for p in papers if p["is_oa"] and p["oa_url"]]
    oa_path = OUTPUT_DIR / "oa_papers_v2.json"
    with open(oa_path, "w") as f:
        json.dump(oa_papers, f, indent=2, default=str)
    print(f"OA papers with URLs saved to {oa_path} ({len(oa_papers)} papers)")

    # Papers with abstracts (ready for RAG)
    rag_ready = [p for p in papers if p["abstract"] and len(p["abstract"]) > 50]
    rag_path = OUTPUT_DIR / "rag_ready_v2.json"
    with open(rag_path, "w") as f:
        json.dump(rag_ready, f, indent=2, default=str)
    print(f"RAG-ready (has abstract): {rag_path} ({len(rag_ready)} papers)")


if __name__ == "__main__":
    main()
