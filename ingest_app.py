"""
audit_ingestion_v04/ingest_app.py
Audit Ingestion Pipeline v04.2 — Streamlit UI

Canonical audit evidence view with extraction diagnostics.
OpenAI only. One model selector. Full provenance display.
"""
import streamlit as st
import json
import sys
import tempfile
import shutil
from pathlib import Path
import pandas as pd

from audit_ingestion.workflow import (
    build_client_followup_package,
    build_prioritized_action_queue,
    compute_bytes_signature,
    merge_state_into_evidence,
    next_best_question,
    persist_evidence_state,
    register_lineage,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

BUILD_VERSION = "v05.1"

st.set_page_config(
    page_title="Audit Ingestion v05",
    page_icon="📋",
    layout="wide",
)

st.markdown("""
<style>
.page-header {
    font-size: 1.4rem; font-weight: 800; color: #1A335C;
    margin-bottom: 0;
}
.section-title {
    font-size: 0.95rem; font-weight: 700; color: #1A335C;
    border-bottom: 2px solid #1A335C;
    padding-bottom: 3px; margin: 16px 0 10px 0;
}
.audit-area-tag {
    background: #dbeafe; color: #1e40af;
    padding: 2px 8px; border-radius: 3px;
    font-size: 0.75rem; font-weight: 600;
    display: inline-block; margin: 2px;
}
.assertion-tag {
    background: #dcfce7; color: #166534;
    padding: 2px 8px; border-radius: 3px;
    font-size: 0.75rem; font-weight: 600;
    display: inline-block; margin: 2px;
}
.match-target-tag {
    background: #f3e8ff; color: #6b21a8;
    padding: 2px 8px; border-radius: 3px;
    font-size: 0.75rem; font-weight: 600;
    display: inline-block; margin: 2px;
}
.claim-box {
    background: #f0fdf4; border-left: 3px solid #16a34a;
    padding: 8px 12px; margin: 6px 0; border-radius: 0 4px 4px 0;
}
.flag-info     { background: #eff6ff; border-left: 3px solid #3b82f6; padding: 8px 12px; margin: 4px 0; border-radius: 0 4px 4px 0; }
.flag-warning  { background: #fffbeb; border-left: 3px solid #f59e0b; padding: 8px 12px; margin: 4px 0; border-radius: 0 4px 4px 0; }
.flag-critical { background: #fef2f2; border-left: 3px solid #dc2626; padding: 8px 12px; margin: 4px 0; border-radius: 0 4px 4px 0; }
.diag-row { font-size: 0.82rem; color: #374151; }
.prov-quote { font-style: italic; color: #6b7280; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


def conf_badge(c: float) -> str:
    if c >= 0.80: color = "#16a34a"
    elif c >= 0.50: color = "#d97706"
    else: color = "#dc2626"
    return f'<span style="color:{color};font-weight:700">{c:.2f}</span>'


def build_diagnostic_csv(raw_results: list, timings: dict) -> str:
    """Full diagnostic CSV — one row per file, every key field flattened."""
    import csv, io
    from audit_ingestion.models import AuditEvidence
    rows = []
    for r in raw_results:
        ev_data = r.get("evidence") or {}
        fname   = ev_data.get("source_file", "?")
        status  = r.get("status", "?").upper()
        try:
            ev   = AuditEvidence(**ev_data)
            # Ensure readiness is always computed — recompute if missing
            # so the export reflects current state even for cached results
            if ev.readiness is None:
                from audit_ingestion.readiness import apply_readiness
                apply_readiness(ev)
            meta = ev.extraction_meta
            ov   = ev.audit_overview
            lk   = ev.link_keys
            ds   = ev_data.get("document_specific") or {}
            seg  = ds.get("_segmentation") or {}
            t    = ds.get("_stage_timings") or {}
            fin  = ds.get("_financial") or {}
            pf = {}
            for i, p in enumerate(ev.parties[:5]):
                pf[f"party_{i+1}_role"] = p.role
                pf[f"party_{i+1}_name"] = p.name
            af = {}
            for i, a in enumerate(ev.amounts[:12]):
                pr = a.provenance
                af[f"amount_{i+1}_type"]  = a.type
                af[f"amount_{i+1}_value"] = a.value
                af[f"amount_{i+1}_page"]  = pr.page if pr else ""
                af[f"amount_{i+1}_conf"]  = f"{pr.confidence:.2f}" if pr else ""
            df2 = {}
            for i, d in enumerate(ev.dates[:6]):
                pr = d.provenance
                df2[f"date_{i+1}_type"]  = d.type
                df2[f"date_{i+1}_value"] = d.value
                df2[f"date_{i+1}_page"]  = pr.page if pr else ""
            idf = {}
            for i, ident in enumerate(ev.identifiers[:6]):
                pr = ident.provenance
                idf[f"id_{i+1}_type"]  = ident.type
                idf[f"id_{i+1}_value"] = ident.value
                idf[f"id_{i+1}_page"]  = pr.page if pr else ""
            all_flags = ev.flags or []
            cf = {}
            for i, c in enumerate((ev.claims or [])[:5]):
                cf[f"claim_{i+1}"] = c.statement
            attachments = "; ".join(
                f"{a.get('name','')} (pp {a['pages'][0]}-{a['pages'][-1]})"
                for a in seg.get("attachments", []) if a.get("pages")
            )
            row = {
                "file": fname, "status": status,
                "family": ev.family.value if ev.family else "",
                "subtype": ev.subtype or "", "title": ev.title or "",
                "summary": ov.summary[:300] if ov else "",
                "audit_areas": "; ".join(ov.audit_areas) if ov else "",
                "assertions": "; ".join(ov.assertions) if ov else "",
                "period": (
                    ov.period.effective_date or
                    f"{ov.period.start or ''} - {ov.period.end or ''}"
                    if ov and ov.period else ""
                ),
                "extractor": meta.primary_extractor,
                "engine_chain": " → ".join(r.get("engine_chain", [])),
                "total_chars": meta.total_chars,
                "pages": meta.pages_processed,
                "weak_pages": meta.weak_pages_count,
                "ocr_pages": meta.ocr_pages_count,
                "vision_pages": meta.vision_pages_count,
                "confidence": meta.overall_confidence,
                "needs_review": meta.needs_human_review,
                "readiness_status": ev.readiness.readiness_status if ev.readiness else "",
                "blocking_state": ev.readiness.blocking_state if ev.readiness else "",
                "blocking_issues": "; ".join(ev.readiness.blocking_issues) if ev.readiness else "",
                "question_count": len(ev.readiness.questions) if ev.readiness else 0,
                "reviewer_questions": "; ".join(q.question_text[:80] for q in (ev.readiness.questions or []) if q.audience == "reviewer" and not q.resolved)[:300] if ev.readiness else "",
                "client_questions": "; ".join(q.question_text[:80] for q in (ev.readiness.questions or []) if q.audience == "client" and not q.resolved)[:300] if ev.readiness else "",
                "population_ready": ev.readiness.population_ready if ev.readiness else "",
                "population_status": ev.readiness.population_status if ev.readiness else "",
                "row_count":           (fin.get("row_diagnostics") or {}).get("row_count", ""),
                "flagged_row_count":   (fin.get("row_diagnostics") or {}).get("flagged_row_count", ""),
                "duplicate_rows":      (fin.get("row_diagnostics") or {}).get("duplicate_rows", ""),
                "malformed_rows":      (fin.get("row_diagnostics") or {}).get("malformed_rows", ""),
                "outlier_rows":        (fin.get("row_diagnostics") or {}).get("outlier_rows", ""),
                "pop_blocking_reasons": "; ".join((fin.get("row_diagnostics") or {}).get("blocking_reasons", [])),
                "open_blocking_questions": sum(1 for q in (ev.readiness.questions if ev.readiness else []) if q.blocking and not q.resolved),
                "resolved_questions":  sum(1 for q in (ev.readiness.questions if ev.readiness else []) if q.resolved),
                "open_client_questions": sum(1 for q in (ev.readiness.questions if ev.readiness else []) if q.audience == "client" and not q.resolved),
                "time_extraction_s": t.get("extraction", ""),
                "time_segmentation_s": t.get("segmentation", ""),
                "time_canonical_s": t.get("canonical_ai", ""),
                "time_rescue_s": t.get("rescue", ""),
                "time_total_s": timings.get(fname, t.get("total", "")),
                "flag_all_types": "; ".join(f.type for f in all_flags),
                "flag_warnings": "; ".join(f"{f.type}: {f.description}" for f in all_flags if f.severity == "warning"),
                "flag_criticals": "; ".join(f"{f.type}: {f.description}" for f in all_flags if f.severity == "critical"),
                "link_party_names":        "; ".join(lk.party_names)        if lk and lk.party_names        else "",
                "link_doc_numbers":         "; ".join(lk.document_numbers)     if lk and lk.document_numbers    else "",
                "link_invoice_numbers":     "; ".join(lk.invoice_numbers)      if lk and lk.invoice_numbers     else "",
                "link_agreement_numbers":   "; ".join(lk.agreement_numbers)    if lk and lk.agreement_numbers   else "",
                "link_recurring_amounts":   "; ".join(str(a) for a in (lk.recurring_amounts or [])) if lk else "",
                "link_key_dates":           "; ".join(lk.key_dates)            if lk and lk.key_dates           else "",
                "link_other_ids":           "; ".join(lk.other_ids)            if lk and lk.other_ids           else "",
                "link_asset_descriptions":  "; ".join(lk.asset_descriptions)   if lk and lk.asset_descriptions  else "",
                "bundle_detected": seg.get("bundle_detected", False),
                "bundle_confidence_band": seg.get("confidence_band", ""),
                "primary_component": seg.get("primary_description", ""),
                "primary_pages": str(seg.get("primary_pages", "")),
                "supporting_components": attachments,
            }
            row.update(pf); row.update(af); row.update(df2)
            row.update(idf); row.update(cf)
        except Exception as exc:
            row = {"file": fname, "status": status,
                   "flag_criticals": f"CSV build error: {exc}"}
        rows.append(row)
    if not rows:
        return ""
    all_keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=all_keys, extrasaction="ignore")
    w.writeheader()
    for row in rows:
        w.writerow(row)
    return buf.getvalue()



def _process_one(args):
    """Worker function for concurrent file processing."""
    import time
    path, api_key, mode, model, allow_rescue = args
    t0 = time.time()
    try:
        from audit_ingestion.router import ingest_one
        result = ingest_one(path, api_key=api_key, model=model, mode=mode,
                            allow_rescue=allow_rescue)
        elapsed = round(time.time() - t0, 2)
        return result, elapsed, None
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        return None, elapsed, str(e)


def run_pipeline(uploaded_files, api_key, mode="fast", allow_rescue=False):
    """
    Run pipeline with concurrent file workers.
    File workers: 4 (conservative)
    AI concurrency implicitly capped by worker count.
    """
    import time
    import concurrent.futures
    from audit_ingestion.providers.openai_provider import DEFAULT_MODEL
    from audit_ingestion.models import IngestionResult, AuditEvidence, Flag

    FILE_WORKERS = min(4, len(uploaded_files))
    batch_start = time.time()

    progress    = st.progress(0)
    status_text = st.empty()
    stage_area  = st.empty()

    tmpdir = tempfile.mkdtemp(prefix="audit_v05_")
    Path(tmpdir, ".tmp").mkdir(exist_ok=True)

    # Write all files first
    file_paths = []
    for uf in uploaded_files:
        file_bytes_data = uf.getvalue()
        tmp_path = Path(tmpdir) / uf.name
        with open(tmp_path, "wb") as f:
            f.write(file_bytes_data)
        # Store bytes so retry can re-process without re-upload
        st.session_state[f"_upload_bytes_{uf.name}"] = file_bytes_data
        file_paths.append(str(tmp_path))

    results = [None] * len(file_paths)
    timings: dict[str, float] = {}
    completed = 0

    try:
        args_list = [
            (path, api_key, mode, DEFAULT_MODEL, allow_rescue)
            for path in file_paths
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=FILE_WORKERS) as executor:
            future_to_idx = {
                executor.submit(_process_one, args): i
                for i, args in enumerate(args_list)
            }

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                fname = Path(file_paths[idx]).name
                completed += 1
                progress.progress(completed / len(file_paths))

                try:
                    result, elapsed, err = future.result()
                    timings[fname] = elapsed
                    if result is not None:
                        results[idx] = result
                        stage = result.engine_chain[-1] if result.engine_chain else "?"
                        status_text.text(
                            f"✅ {fname} — {elapsed}s | chain: {' → '.join(result.engine_chain)}"
                        )
                    else:
                        results[idx] = IngestionResult(
                            status="failed",
                            errors=[err or "Unknown error"],
                            evidence=AuditEvidence(
                                source_file=fname,
                                flags=[Flag(type="fatal_error",
                                           description=err or "Unknown", severity="critical")]
                            ),
                        )
                        status_text.text(f"❌ {fname} failed — {elapsed}s")
                except Exception as e:
                    results[idx] = IngestionResult(
                        status="failed",
                        errors=[str(e)],
                        evidence=AuditEvidence(
                            source_file=fname,
                            flags=[Flag(type="fatal_error",
                                       description=str(e), severity="critical")]
                        ),
                    )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        progress.empty()
        status_text.empty()

    batch_elapsed = round(time.time() - batch_start, 2)
    st.caption(
        f"Batch complete — {len(results)} files in {batch_elapsed}s "
        f"({FILE_WORKERS} workers | mode: {mode})"
    )

    return results, timings


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # API Key
    key_file = Path("openai_key.txt")
    default_key = key_file.read_text().strip() if key_file.exists() else ""
    api_key = st.text_input(
        "OpenAI API Key",
        value=default_key, type="password", placeholder="sk-...",
        key="api_key_input",
    )
    if api_key:
        st.success("✅ Key ready")
    else:
        st.warning("⚠️ API key required for extraction")

    # Processing Mode
    processing_mode = st.radio(
        "Processing Mode",
        ["Fast Review", "Deep Extraction"],
        index=0,
        help=(
            "Fast Review: pdfplumber + PyPDF2 + auto-escalation. Quick.\n"
            "Deep Extraction: adds OCR + vision on weak pages."
        ),
    )
    mode = "fast" if processing_mode == "Fast Review" else "deep"

    allow_rescue = st.checkbox(
        "Allow gpt-5.4-pro rescue on worst page(s)",
        value=False,
        help=(
            "Off by default. When enabled, gpt-5.4-pro may be used for freeform "
            "rescue on pages that canonical extraction cannot read. "
            "NOT used for canonical structured JSON."
        ),
    )
    st.caption(f"Model: `gpt-5.4` | Mode: `{mode}`")
    st.caption(f"Build: `{BUILD_VERSION}`")

    st.markdown("---")
    st.markdown("### 📋 v05 Architecture")
    st.markdown("""
1. **Page-aware extraction**
   pdfplumber → PyPDF2 → extractous → OCR → vision
2. **Canonical AI pass**
   Single structured JSON extraction
3. **Normalization**
   Parties, dates, amounts, link keys
4. **Audit evidence object**
   Facts + Claims + Flags + Link Keys
    """)
    st.caption("Audit Ingestion Pipeline v05")


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-header">📋 Audit Ingestion Pipeline</div>', unsafe_allow_html=True)
st.markdown("Upload any audit document — lease, invoice, grant, bank statement, minutes, payroll, or any other.")
st.markdown("---")

uploaded_files = st.file_uploader(
    "Drop documents here",
    accept_multiple_files=True,
    type=["pdf", "csv", "xlsx", "xls", "txt", "docx"],
    label_visibility="visible",
)

# Store uploaded file references AND bytes in session state for per-file retry.
# Bytes are stored separately because the uploader widget resets on rerun
# but bytes persist in session state until explicitly cleared.
if uploaded_files:
    st.session_state["uploaded_files_ref"] = uploaded_files
    for _uf in uploaded_files:
        st.session_state[f"_upload_bytes_{_uf.name}"] = _uf.getvalue()

c1, c2 = st.columns([2, 1])
with c1:
    run_btn = st.button("▶ Run Pipeline", type="primary",
                        disabled=not uploaded_files or not api_key)
with c2:
    if st.button("🗑 Clear"):
        st.session_state.pop("v05_results", None)
        st.rerun()

if not api_key and uploaded_files:
    st.warning("Enter your OpenAI API key in the sidebar to run extraction.")

if run_btn and uploaded_files and api_key:
    with st.spinner("Running page-aware extraction and canonical AI analysis..."):
        results, timings = run_pipeline(uploaded_files, api_key, mode, allow_rescue)
    _prior_results = {((r.get("evidence") or {}).get("source_file")): r for r in st.session_state.get("v05_results", [])}
    _result_payloads = []
    for _res in results:
        _payload = _res.model_dump()
        _ev_data = _payload.get("evidence") or {}
        _fname = _ev_data.get("source_file")
        if _fname:
            try:
                from audit_ingestion.models import AuditEvidence
                _ev = AuditEvidence(**_ev_data)
                _bytes = st.session_state.get(f"_upload_bytes_{_fname}", b"")
                _sig = compute_bytes_signature(_bytes) if _bytes else _fname
                merge_state_into_evidence(_ev, file_signature=_sig)
                _prior_payload = _prior_results.get(_fname)
                _prior_ev = AuditEvidence(**((_prior_payload or {}).get("evidence") or {})) if _prior_payload else None
                if _prior_ev:
                    register_lineage(_ev, _prior_ev)
                persist_evidence_state(_ev)
                _payload["evidence"] = _ev.model_dump()
            except Exception:
                pass
        _result_payloads.append(_payload)
    st.session_state["v05_results"] = _result_payloads
    st.session_state["v05_timings"] = timings
    st.rerun()


# ── Results ───────────────────────────────────────────────────────────────────
if "v05_results" not in st.session_state:
    st.stop()

raw_results = st.session_state["v05_results"]
for _idx, _result in enumerate(raw_results):
    _ev_data = _result.get("evidence") or {}
    _fname = _ev_data.get("source_file")
    if not _fname:
        continue
    try:
        from audit_ingestion.models import AuditEvidence
        _ev = AuditEvidence(**_ev_data)
        _bytes = st.session_state.get(f"_upload_bytes_{_fname}", b"")
        _sig = compute_bytes_signature(_bytes) if _bytes else _fname
        merge_state_into_evidence(_ev, file_signature=_sig)
        raw_results[_idx]["evidence"] = _ev.model_dump()
    except Exception:
        pass


def _update_evidence_in_session(source_file: str, updater):
    from audit_ingestion.models import AuditEvidence
    for idx, result in enumerate(st.session_state.get("v05_results", [])):
        ev_data = result.get("evidence") or {}
        if ev_data.get("source_file") != source_file:
            continue
        ev = AuditEvidence(**ev_data)
        updater(ev)
        persist_evidence_state(ev)
        st.session_state["v05_results"][idx]["evidence"] = ev.model_dump()


def _question_detail(ev, q):
    _matching_flag = next((f for f in (ev.flags or []) if f.type == q.source_flag), None)
    return ((_matching_flag.description if _matching_flag else "") or q.question_text).strip()


def _matching_bulk_targets(source_file: str, question_type: str, source_flag: str | None):
    from audit_ingestion.models import AuditEvidence
    targets = []
    for result in st.session_state.get("v05_results", []):
        ev_data = result.get("evidence") or {}
        fname = ev_data.get("source_file")
        if not fname or fname == source_file:
            continue
        try:
            ev = AuditEvidence(**ev_data)
        except Exception:
            continue
        rd = ev.readiness
        if not rd:
            continue
        for q in rd.questions or []:
            if q.resolved:
                continue
            if q.question_type == question_type and q.source_flag == source_flag:
                targets.append(fname)
                break
    return sorted(set(targets))


def _resolve_matching_questions_bulk(source_files, question_type: str, source_flag: str | None, resolution: str, resolution_type: str):
    from audit_ingestion.readiness import resolve_question as _resolve_question_inline
    from audit_ingestion.models import AuditEvidence
    for fname in source_files:
        for idx, result in enumerate(st.session_state.get("v05_results", [])):
            ev_data = result.get("evidence") or {}
            if ev_data.get("source_file") != fname:
                continue
            ev = AuditEvidence(**ev_data)
            rd = ev.readiness
            if not rd:
                continue
            target_q = next((q for q in (rd.questions or []) if (not q.resolved) and q.question_type == question_type and q.source_flag == source_flag), None)
            if not target_q:
                continue
            _resolve_question_inline(
                ev,
                target_q.question_id,
                resolution or "resolved in bulk",
                actor="reviewer",
                resolution_type=resolution_type,
                comment=resolution or "resolved in bulk",
            )
            persist_evidence_state(ev)
            st.session_state["v05_results"][idx]["evidence"] = ev.model_dump()
            break


def _focus_document(source_file: str, question_id: str | None = None):
    st.session_state["detail_selected"] = source_file
    if question_id:
        st.session_state["_focus_question_id"] = question_id


def _save_financial_override(fname: str, fin_data: dict):
    """Persist a financial override into session state results so rerun picks it up."""
    for _sr in st.session_state.get("v05_results", []):
        _sr_fname = (_sr.get("evidence") or {}).get("source_file", "")
        if _sr_fname == fname:
            (_sr.get("evidence") or {}).setdefault("document_specific", {})["_financial"] = fin_data
            break
        return True
    return False

# Metrics
total   = len(raw_results)
success = sum(1 for r in raw_results if r["status"] == "success")
partial = sum(1 for r in raw_results if r["status"] == "partial")
failed  = sum(1 for r in raw_results if r["status"] == "failed")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total", total)
m2.metric("✅ Success", success)
m3.metric("⚠️ Partial", partial)
m4.metric("❌ Failed", failed)

st.markdown("---")

# Light question guidance — use the summary badge to open file-specific questions.
from audit_ingestion.readiness import resolve_question
_evidence_items_for_queue = []
for _r in raw_results:
    _ev_data = _r.get("evidence") or {}
    try:
        _evidence_items_for_queue.append(AuditEvidence(**_ev_data))
    except Exception:
        pass
_next_best = next_best_question(_evidence_items_for_queue)
if _next_best:
    _best_detail = _next_best.get("flag_description") or _next_best["question_text"]
    st.info(f"Next best item: {_next_best['source_file']} — {_best_detail}")
    st.caption("Click the yellow question badge in Document Summary to answer only the questions for that file.")
if st.session_state.get("_opened_from_queue"):
    st.success(f"Opened {st.session_state['_opened_from_queue']} below in Document Detail.")
    st.session_state.pop("_opened_from_queue")
st.markdown("---")

# Summary table
st.markdown('<div class="section-title">Document Summary</div>', unsafe_allow_html=True)

from audit_ingestion.legacy import canonical_summary_row
from audit_ingestion.models import AuditEvidence

timings = st.session_state.get("v05_timings", {})
import os as _os
_cache_dir = _os.path.join(_os.path.dirname(__file__), ".canonical_cache")

summary_rows = []
for r in raw_results:
    ev_data = r.get("evidence") or {}
    try:
        ev  = AuditEvidence(**ev_data)
        # Recompute readiness if missing (cached results from before readiness was built)
        if ev.readiness is None:
            from audit_ingestion.readiness import apply_readiness
            apply_readiness(ev)
        row = canonical_summary_row(ev)
        row["status"] = r["status"].upper()
        rd = ev.readiness
        row["readiness"] = rd.readiness_status if rd else ""
        row["blocking"]  = (rd.blocking_state == "blocking") if rd else False
        row["q_count"]   = sum(1 for q in (rd.questions or []) if not q.resolved) if rd else 0
        summary_rows.append(row)
    except Exception:
        summary_rows.append({
            "file": ev_data.get("source_file", "?"),
            "status": r["status"].upper(),
            "family": "?", "summary": "Parse error",
            "primary_party": "—", "primary_amount": "—",
            "audit_areas": "—", "confidence": "0.00",
            "extractor": "—", "chars": 0, "needs_review": True,
        })

for row in summary_rows:
    row["time_s"] = f"{timings.get(row.get('file',''), 0):.1f}s"

# ── Summary table with per-row checkboxes ─────────────────────────────────────
# Render one row at a time using columns so we can embed a checkbox per row

_STATUS_ICON = {"SUCCESS": "✅", "PARTIAL": "⚠️", "FAILED": "❌"}
_STATUS_BG   = {"SUCCESS": "#f0fdf4", "PARTIAL": "#fffbeb", "FAILED": "#fef2f2"}

# Header row
_hc = st.columns([0.3, 2.5, 0.8, 1.0, 1.0, 1.4, 1.2, 0.8, 0.4])
for col, label in zip(_hc, ["", "File", "Status", "Family", "Readiness",
                              "Audit Areas", "Confidence", "Questions", ""]):
    col.markdown(f"**{label}**")

# Track checkbox state
if "retry_selected" not in st.session_state:
    st.session_state["retry_selected"] = set()

for row in summary_rows:
    fname  = row.get("file", "?")
    status = row.get("status", "?")
    is_bad = status in ("FAILED", "PARTIAL")
    bg     = _STATUS_BG.get(status, "#ffffff")
    icon   = _STATUS_ICON.get(status, "—")

    _rc = st.columns([0.3, 2.5, 0.8, 1.0, 1.0, 1.4, 1.2, 0.8, 0.4])
    with _rc[0]:
        checked = st.checkbox("", key=f"chk_{fname}",
                              value=(fname in st.session_state["retry_selected"]),
                              label_visibility="collapsed")
        if checked:
            st.session_state["retry_selected"].add(fname)
        else:
            st.session_state["retry_selected"].discard(fname)
    _rc[1].markdown(f"<small>{fname}</small>", unsafe_allow_html=True)
    _rc[2].markdown(f"{icon} {status}")
    _rc[3].markdown(f"<small>{row.get('family','—')}</small>", unsafe_allow_html=True)
    # Readiness badge
    _rd = row.get("readiness", "")
    _rd_icon = {
        "ready":                        "✅",
        "needs_reviewer_confirmation":  "🔵",
        "needs_client_answer":          "🟡",
        "exception_open":               "🟠",
        "unusable":                     "❌",
        "":                             "—",
    }.get(_rd, "—")
    _rd_label = _rd.replace("_", " ").title() if _rd else "—"
    _rc[4].markdown(f"<small>{_rd_icon} {_rd_label}</small>", unsafe_allow_html=True)
    _rc[5].markdown(f"<small>{row.get('audit_areas','—')}</small>", unsafe_allow_html=True)
    _rc[6].markdown(f"<small>{row.get('confidence','—')}</small>", unsafe_allow_html=True)
    _qc = row.get("q_count", 0)
    with _rc[7]:
        if _qc:
            if st.button(f"⚠️ {_qc} Q", key=f"summary_q_{fname}", help=f"Show open questions for {fname}"):
                st.session_state["_summary_question_file"] = fname
                st.session_state["_summary_question_open"] = True
                st.rerun()
        else:
            st.markdown("<small>—</small>", unsafe_allow_html=True)
    with _rc[8]:
        if st.button("🗑", key=f"del_{fname}", help=f"Remove {fname} from results"):
            _new_results = [
                r for r in st.session_state.get("v05_results", [])
                if (r.get("evidence") or {}).get("source_file") != fname
            ]
            st.session_state["v05_results"] = _new_results
            st.session_state["retry_selected"].discard(fname)
            st.session_state.pop(f"_upload_bytes_{fname}", None)
            st.rerun()

_summary_question_file = st.session_state.get("_summary_question_file")
if _summary_question_file:
    _summary_match = next((x for x in raw_results if (x.get("evidence") or {}).get("source_file") == _summary_question_file), None)
    if _summary_match:
        try:
            _summary_ev = AuditEvidence(**((_summary_match.get("evidence") or {})))
        except Exception:
            _summary_ev = None
        if _summary_ev and _summary_ev.readiness and _summary_ev.readiness.questions:
            _open_qs = [q for q in _summary_ev.readiness.questions if not q.resolved]
            if _open_qs:
                st.markdown("---")
                with st.expander(f"Questions for {_summary_question_file} ({len(_open_qs)})", expanded=st.session_state.get("_summary_question_open", True)):
                    _qq1, _qq2 = st.columns([6, 1])
                    with _qq1:
                        st.caption("Answer questions for this file here, or open the file for full detail.")
                    with _qq2:
                        if st.button("Hide", key=f"hide_summary_questions_{_summary_question_file}"):
                            st.session_state.pop("_summary_question_file", None)
                            st.session_state["_summary_question_open"] = False
                            st.rerun()
                    from audit_ingestion.readiness import resolve_question as _resolve_question_inline
                    for _q in _open_qs:
                        _qid = _q.question_id
                        _detail = _question_detail(_summary_ev, _q)
                        _bulk_targets = _matching_bulk_targets(_summary_question_file, _q.question_type, _q.source_flag)
                        st.markdown(f"**{_q.question_text}**")
                        if _detail and _detail != _q.question_text:
                            st.caption(f"What was found: {_detail}")
                        if _bulk_targets:
                            _default_targets = _bulk_targets if len(_bulk_targets) <= 5 else []
                            _apply_files = st.multiselect(
                                "Also apply to matching files",
                                _bulk_targets,
                                default=_default_targets,
                                key=f"summary_apply_{_summary_question_file}_{_qid}",
                                help="Use this when multiple files raise the same question.",
                            )
                        else:
                            _apply_files = []
                        _sq = st.columns([3.0, 1.2, 0.9, 1])
                        with _sq[0]:
                            _ans = st.text_input(
                                "Answer",
                                key=f"summary_answer_{_summary_question_file}_{_qid}",
                                placeholder="Type your answer or note here...",
                                label_visibility="collapsed",
                            )
                        with _sq[1]:
                            _rtype = st.selectbox(
                                "Resolution type",
                                ["answer", "reviewer_confirmed", "override", "dismissed"],
                                key=f"summary_type_{_summary_question_file}_{_qid}",
                                label_visibility="collapsed",
                            )
                        with _sq[2]:
                            if st.button("Resolve", key=f"summary_resolve_{_summary_question_file}_{_qid}", type="primary"):
                                def _apply_summary_resolution(_ev, _qid=_qid, _ans=_ans, _rtype=_rtype):
                                    _resolve_question_inline(
                                        _ev, _qid,
                                        _ans or "resolved from document summary",
                                        actor="reviewer",
                                        resolution_type=_rtype,
                                        comment=_ans or "resolved from document summary",
                                    )
                                _update_evidence_in_session(_summary_question_file, _apply_summary_resolution)
                                if _apply_files:
                                    _resolve_matching_questions_bulk(_apply_files, _q.question_type, _q.source_flag, _ans or "resolved from document summary", _rtype)
                                _focus_document(_summary_question_file, _qid)
                                st.session_state["_opened_from_queue"] = _summary_question_file
                                st.rerun()
                        with _sq[3]:
                            if st.button("Open file", key=f"summary_open_{_summary_question_file}_{_qid}"):
                                _focus_document(_summary_question_file, _qid)
                                st.session_state["_opened_from_queue"] = _summary_question_file
                                st.rerun()
            else:
                st.session_state.pop("_summary_question_file", None)
                st.session_state["_summary_question_open"] = False

st.markdown("---")

# ── Export buttons ────────────────────────────────────────────────────────────
_exp1, _exp2 = st.columns(2)
with _exp1:
    _sum_df = pd.DataFrame(summary_rows)
    st.download_button(
        "⬇️ Export Summary CSV",
        data=_sum_df.to_csv(index=False),
        file_name="audit_evidence_v05_summary.csv",
        mime="text/csv",
        help="One row per file — key fields only",
    )
with _exp2:
    st.download_button(
        "⬇️ Export Diagnostic CSV",
        data=build_diagnostic_csv(raw_results, timings),
        file_name="audit_evidence_v05_diagnostic.csv",
        mime="text/csv",
        help="Full detail — every extracted field, flags, timings, segmentation",
    )

# ── Retry Controls ───────────────────────────────────────────────────────────
_failed_files  = [r for r in summary_rows if r.get("status") == "FAILED"]
_partial_files = [r for r in summary_rows if r.get("status") == "PARTIAL"]
_needs_retry   = _failed_files + _partial_files
_any_selected  = bool(st.session_state.get("retry_selected"))

# Rerun controls — always shown when there are results
st.markdown("#### 🔄 Rerun Controls")
_r1, _r2, _r3, _r4, _r5 = st.columns([1.4, 1.2, 1.2, 1.2, 2])

with _r1:
    if st.button("▶ Rerun Selected", key="retry_selected_btn",
                 type="primary" if _any_selected else "secondary"):
        if not _any_selected:
            st.warning("Check the boxes next to the files you want to rerun first.")
        else:
            st.session_state["_retry_queue"] = list(st.session_state["retry_selected"])
            st.session_state["_retry_trigger"] = True

with _r2:
    if st.button("Rerun All", key="retry_all_btn"):
        st.session_state["_retry_queue"] = [r["file"] for r in summary_rows]
        st.session_state["_retry_trigger"] = True

with _r3:
    if st.button("Retry All Failed", key="retry_all_failed_btn",
                 disabled=not _failed_files):
        st.session_state["_retry_queue"] = [r["file"] for r in _failed_files]
        st.session_state["_retry_trigger"] = True

with _r4:
    if st.button("Retry All Partial", key="retry_all_partial_btn",
                 disabled=not _partial_files):
        st.session_state["_retry_queue"] = [r["file"] for r in _partial_files]
        st.session_state["_retry_trigger"] = True

with _r5:
    _retry_mode_choice = st.radio(
        "Mode",
        ["Fast Review", "Deep Extraction"],
        index=1,
        horizontal=True,
        key="retry_mode_radio",
    )

# Failed files quick view — only when there are failures
if _needs_retry:
    with st.expander(f"📋 Files needing attention ({len(_needs_retry)})", expanded=False):
        for _nr in _needs_retry:
            _icon = "❌" if _nr.get("status") == "FAILED" else "⚠️"
            st.markdown(f"{_icon} **{_nr['file']}** — {_nr.get('family','?')} | "
                        f"conf: {_nr.get('confidence','?')} | {_nr.get('time_s','')}")

    # ── Execute retry when triggered ─────────────────────────────────────────
    if st.session_state.get("_retry_trigger"):
        st.session_state["_retry_trigger"] = False
        _queue     = st.session_state.pop("_retry_queue", [])
        _mode_key  = "deep" if st.session_state.get("retry_mode_radio") == "Deep Extraction" else "fast"
        _api_key_r = st.session_state.get("api_key_input", "")
        _uf_map    = {uf.name: uf for uf in st.session_state.get("uploaded_files_ref", [])}

        _r_bar = st.progress(0)
        _r_status = st.empty()
        _retry_out = []
        _retry_tim = {}

        for _qi, _qf in enumerate(_queue):
            _r_status.text(f"Retrying {_qf} ({_qi+1}/{len(_queue)})...")
            _r_bar.progress((_qi + 1) / max(len(_queue), 1))
            _uf = _uf_map.get(_qf)
            import tempfile as _tf, time as _ti, os as _tos
            # Fall back to stored bytes if the live uploader reference is gone
            if _uf:
                _fbytes = _uf.getvalue()
            else:
                _fbytes = st.session_state.get(f"_upload_bytes_{_qf}")
            if not _fbytes:
                st.warning(f"⚠️ {_qf} — file bytes not found. Upload the file again then retry.")
                continue
            with _tf.NamedTemporaryFile(
                suffix=_tos.path.splitext(_qf)[1] or ".pdf",
                delete=False, prefix="retry_"
            ) as _tmp:
                _tmp.write(_fbytes)
                _tp = _tmp.name
            _t0 = _ti.perf_counter()
            try:
                from audit_ingestion.router import ingest_one
                _locked       = st.session_state.get("_retry_locked_type", {})
                _override_type = _locked.get(_qf)
                _sheet_lock   = st.session_state.get("_retry_excel_sheet", {})
                _override_sheet = _sheet_lock.get(_qf)

                # Clear the disk cache entry for this file so the retry
                # forces a fresh AI call instead of returning the cached result
                try:
                    import hashlib, os as _cache_os
                    _cache_dir_r = _cache_os.path.join(
                        _cache_os.path.dirname(__file__), ".canonical_cache"
                    )
                    if _cache_os.path.isdir(_cache_dir_r):
                        for _cf in _cache_os.listdir(_cache_dir_r):
                            if _cf.endswith(".json"):
                                _cache_os.unlink(_cache_os.path.join(_cache_dir_r, _cf))
                                # Only need to clear — ingest_one will repopulate
                                break  # Clear all entries to be safe on retry
                        # Actually clear ALL entries since we want fresh results
                        import shutil as _retry_shutil
                        _retry_shutil.rmtree(_cache_dir_r, ignore_errors=True)
                except Exception:
                    pass

                _res = ingest_one(
                    _tp, api_key=_api_key_r, mode=_mode_key,
                    allow_rescue=st.session_state.get("allow_rescue", False),
                    financial_type_override=_override_type,
                    financial_sheet_override=_override_sheet,
                    bypass_cache=True,
                )
                _ev_dump = _res.evidence.model_dump() if _res.evidence else {}
                # Force source_file back to the original filename — the temp path must not leak
                if _ev_dump:
                    _ev_dump["source_file"] = _qf
                _retry_out.append({
                    "status": _res.status,
                    "evidence": _ev_dump,
                    "errors": _res.errors, "engine_chain": _res.engine_chain,
                })
            except Exception as _ex:
                _retry_out.append({
                    "status": "failed",
                    "evidence": {"source_file": _qf},
                    "errors": [str(_ex)], "engine_chain": [],
                })
            finally:
                try: _tos.unlink(_tp)
                except: pass
            _retry_tim[_qf] = round(_ti.perf_counter() - _t0, 1)

        _r_bar.empty()
        _r_status.empty()

        if _retry_out:
            def _get_fname(r):
                # Results from original pipeline: file is in r["evidence"]["source_file"]
                # Results from retry: file is also in r["evidence"]["source_file"]
                return (r.get("evidence") or {}).get("source_file", r.get("file", "?"))
            _existing = {_get_fname(r): r for r in st.session_state.get("v05_results", [])}
            for _r in _retry_out:
                _existing[_get_fname(_r)] = _r
            st.session_state["v05_results"] = list(_existing.values())
            _mt = dict(st.session_state.get("v05_timings", {}))
            _mt.update(_retry_tim)
            st.session_state["v05_timings"] = _mt
            st.session_state["retry_selected"] = set()
            st.success(f"✅ Retry complete — {len(_retry_out)} file(s) reprocessed.")
            st.rerun()

st.markdown("---")

# Cache stats + clear button
_cache_files = []
if _os.path.isdir(_cache_dir):
    _cache_files = [f for f in _os.listdir(_cache_dir) if f.endswith(".json")]

# Warn if any files are Unusable — most common cause is a stale cache entry
_unusable_files = [r for r in summary_rows if r.get("readiness") == "unusable"]
if _unusable_files:
    _u_names = ", ".join(r["file"] for r in _unusable_files[:3])
    st.warning(
        f"⚠️ **{len(_unusable_files)} file(s) showing Unusable** ({_u_names}). "
        f"This is usually caused by a stale cache entry from a previous version. "
        f"Click **Clear Cache** below and re-run those files."
    )

_cc1, _cc2, _cc3 = st.columns([2, 2, 1])
_cc1.caption(f"💾 Disk cache: **{len(_cache_files)}** result(s) stored")
_cc2.caption("Re-uploading the same files will skip OpenAI and use cache.")
with _cc3:
    if st.button("🗑 Clear Cache", help="Delete all cached canonical results"):
        import shutil as _shutil
        if _os.path.isdir(_cache_dir):
            _shutil.rmtree(_cache_dir)
        st.success("Cache cleared.")
        st.rerun()

st.markdown("---")

# File detail selector
st.markdown('<div class="section-title">Document Detail</div>', unsafe_allow_html=True)
file_names = [r.get("evidence", {}).get("source_file", f"File {i}")
              for i, r in enumerate(raw_results)]
if "detail_selected" not in st.session_state or st.session_state["detail_selected"] not in file_names:
    st.session_state["detail_selected"] = file_names[0] if file_names else None
selected = st.selectbox("Select document to inspect", file_names, key="detail_selected")
if st.session_state.get("_focus_question_id"):
    st.caption("Focused from Questions to Resolve")

r = next((x for x in raw_results
          if x.get("evidence", {}).get("source_file") == selected), None)
if not r:
    st.stop()

ev_data = r.get("evidence") or {}
try:
    ev = AuditEvidence(**ev_data)
except Exception as e:
    st.error(f"Could not parse evidence: {e}")
    st.stop()

meta = ev.extraction_meta
overview = ev.audit_overview

# ── Section 1: Auditor Snapshot ───────────────────────────────────────────────
if overview:
    st.markdown('<div class="section-title">🔍 Auditor Snapshot</div>', unsafe_allow_html=True)

    family = ev.family.value.replace("_", " ").title()
    subtype = ev.subtype or "—"
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Document Family:** `{family}`")
    c2.markdown(f"**Subtype:** `{subtype}`")
    c3.markdown(f"**Title:** {ev.title or '—'}")

    st.markdown(f"**Summary:** {overview.summary}")

    if overview.audit_areas:
        tags = " ".join(f'<span class="audit-area-tag">{a}</span>'
                        for a in overview.audit_areas)
        st.markdown(f"**Audit Areas:** {tags}", unsafe_allow_html=True)

    if overview.assertions:
        tags = " ".join(f'<span class="assertion-tag">{a}</span>'
                        for a in overview.assertions)
        st.markdown(f"**Assertions:** {tags}", unsafe_allow_html=True)

    if overview.period:
        p = overview.period
        period_parts = []
        if p.effective_date: period_parts.append(f"Effective: {p.effective_date}")
        if p.start:          period_parts.append(f"Start: {p.start}")
        if p.end:            period_parts.append(f"End: {p.end}")
        if p.term_months:    period_parts.append(f"Term: {p.term_months} months")
        if period_parts:
            st.markdown(f"**Period:** {' | '.join(period_parts)}")

    if overview.match_targets:
        tags = " ".join(f'<span class="match-target-tag">{t}</span>'
                        for t in overview.match_targets)
        st.markdown(f"**Match Targets:** {tags}", unsafe_allow_html=True)

    # ── v05 Bundle / Segmentation Summary ─────────────────────────────────────
    seg = doc_specific.get("_segmentation")
    if seg and seg.get("bundle_detected"):
        st.markdown("---")
        primary_desc = seg.get("primary_description", "")
        st.markdown(f"🗂 **Primary document identified:** {primary_desc}")
        attachments = seg.get("attachments", [])
        if attachments:
            st.markdown("📎 **Supporting attachments separated:**")
            for a in attachments:
                pages = a.get("pages", [])
                pg_str = f"pages {pages[0]}–{pages[-1]}" if pages else ""
                ids = a.get("key_identifiers", [])
                id_str = f" — {', '.join(ids[:3])}" if ids else ""
                st.markdown(
                    f"  - **{a['name']}** ({pg_str}): "
                    f"{a.get('summary', '')}{id_str}"
                )
        if seg.get("confidence_band") in ("medium", "low"):
            st.caption("📌 Extracted conservatively — document may contain additional sections.")
    elif seg and not seg.get("bundle_detected") and seg.get("confidence_band") == "low":
        st.caption("📌 Document may contain multiple sections; extracted conservatively.")

    # ── Financial file panel ──────────────────────────────────────────────────
    _fin = doc_specific.get("_financial")
    if _fin and _fin.get("doc_type") and _fin["doc_type"] != "not_financial_structured_data":
        st.markdown("---")

        _ftype   = _fin["doc_type"]
        _fconf   = float(_fin.get("doc_type_confidence", 0))
        _ffin    = _fin.get("finality_state", "")
        _fsource = _fin.get("doc_type_source", "")
        _fperiod_start = _fin.get("period_start")
        _fperiod_end   = _fin.get("period_end", "")
        _fperiod_conf  = float(_fin.get("period_confidence", 0))
        _fbal          = _fin.get("balance_check", {})
        _ftotals       = _fin.get("totals", {})
        _sheet_count   = int(_fin.get("excel_sheet_count") or 1)
        _sheet_name    = _fin.get("excel_sheet_name")
        _all_sheets    = _fin.get("excel_all_sheets", [])

        _TYPE_LABELS = {
            "general_ledger":             "General Ledger",
            "journal_entry_listing":      "Journal Entry Listing",
            "trial_balance_current":      "Trial Balance — Current Year",
            "trial_balance_prior_year":   "Trial Balance — Prior Year",
            "trial_balance_unknown_year": "Trial Balance — Year Unknown",
            "budget":                     "Budget",
            "bank_statement_csv":         "Bank Statement (CSV)",
            "chart_of_accounts":          "Chart of Accounts",
        }
        _ALL_TYPES = list(_TYPE_LABELS.keys())

        _SOURCE_LABELS = {
            "heuristic":    "column headers",
            "ai":           "AI fallback",
            "user_override":"user confirmed",
            "filename":     "filename",
        }

        # ── Header: awaiting confirmation vs confirmed/trusted ────────────────
        if _ffin == "review_required":
            st.markdown(
                "<div style='background:#fffbeb;border-left:4px solid #f59e0b;"
                "padding:14px 16px;border-radius:0 6px 6px 0;margin-bottom:12px'>"
                "<span style='font-size:1.05rem;font-weight:700;color:#92400e'"
                ">⏳ Awaiting Classification Confirmation</span><br>"
                "<span style='color:#78350f;font-size:0.88rem'>"
                "This file needs one decision before extraction can proceed. "
                "Review the detected type below and confirm or correct it.</span></div>",
                unsafe_allow_html=True
            )
        elif _ffin == "review_recommended":
            st.markdown(
                "<div style='background:#eff6ff;border-left:4px solid #3b82f6;"
                "padding:10px 16px;border-radius:0 6px 6px 0;margin-bottom:12px'>"
                "<span style='font-weight:700;color:#1e40af'>🟡 Review Recommended</span> "
                "<span style='color:#1e40af;font-size:0.88rem'>"
                "— Classification confidence is moderate. Verify the type is correct.</span></div>",
                unsafe_allow_html=True
            )
        elif _ffin == "user_confirmed":
            st.markdown(
                "<div style='background:#f0fdf4;border-left:4px solid #16a34a;"
                "padding:8px 16px;border-radius:0 6px 6px 0;margin-bottom:10px'>"
                "<span style='font-weight:700;color:#15803d'>✅ Classification Confirmed</span></div>",
                unsafe_allow_html=True
            )

        # ── Step 1: Detected Type ─────────────────────────────────────────────
        st.markdown("**Document Type**")
        _conf_color = "#16a34a" if _fconf >= 0.85 else ("#d97706" if _fconf >= 0.70 else "#dc2626")
        _c1, _c2 = st.columns([3, 2])
        _c1.markdown(
            f"<span style='font-size:1.1rem;font-weight:700'>{_TYPE_LABELS.get(_ftype, _ftype)}</span>  "
            f"<span style='color:{_conf_color};font-weight:600'>{_fconf:.0%} confidence</span>  "
            f"<span style='color:#9ca3af;font-size:0.82rem'>via {_SOURCE_LABELS.get(_fsource, _fsource)}</span>",
            unsafe_allow_html=True
        )

        # ── Step 2: Period ────────────────────────────────────────────────────
        st.markdown("**Period Covered**")
        if _fperiod_start:
            _period_str = _fperiod_start if (_fperiod_start == _fperiod_end or not _fperiod_end)                           else f"{_fperiod_start} – {_fperiod_end}"
            _psrc = _fin.get("period_source", "")
            _psrc_label = {"header": "explicit header", "filename": "filename",
                           "data_inferred": "inferred from dates"}.get(_psrc, _psrc)
            st.markdown(
                f"**{_period_str}** &nbsp; "
                f"<span style='color:#6b7280;font-size:0.82rem'>{_fperiod_conf:.0%} confidence ({_psrc_label})</span>",
                unsafe_allow_html=True
            )
        else:
            st.warning("Period not detected — check filename or file header for fiscal year information.")

        # ── Step 3: Sheet (Excel only) ────────────────────────────────────────
        if _sheet_count > 1:
            st.markdown("**Workbook Sheet**")
            _sh_cols = st.columns([3, 2])
            _sh_cols[0].markdown(
                f"Using sheet: **{_sheet_name}** &nbsp; "
                f"<span style='color:#6b7280;font-size:0.82rem'>({_sheet_count} sheets in workbook)</span>",
                unsafe_allow_html=True
            )
            _other_sheets = [s for s in _all_sheets if s != _sheet_name]
            if _other_sheets:
                _chosen_sheet = _sh_cols[1].selectbox(
                    "Switch to sheet:",
                    options=[""] + _all_sheets,
                    index=0,
                    key=f"sheet_select_{selected}",
                    label_visibility="collapsed",
                )
                if _chosen_sheet and _chosen_sheet != _sheet_name:
                    if st.button(f"↩ Use '{_chosen_sheet}'", key=f"sheet_confirm_{selected}"):
                        st.session_state["_retry_queue"] = [selected]
                        st.session_state["_retry_trigger"] = True
                        st.session_state["_retry_excel_sheet"] = {selected: _chosen_sheet}
                        st.rerun()

        # ── Step 4: Key Totals ────────────────────────────────────────────────
        _display_totals = {k: v for k, v in _ftotals.items()
                           if not k.endswith("_error") and isinstance(v, (int, float))}
        if _display_totals:
            st.markdown("**Key Totals**")
            _tcols = st.columns(min(4, len(_display_totals)))
            for _ti, (_tk, _tv) in enumerate(list(_display_totals.items())[:4]):
                _tcols[_ti].metric(_tk.replace("_", " ").title(), f"${_tv:,.0f}")

        # Balance check alert
        _bal_flag = _fbal.get("flag_level", "")
        if _bal_flag == "material_balance_difference":
            st.error(
                f"⚠️ **Out of Balance** — "
                f"DR ${_fbal.get('dr_total', 0):,.2f} vs CR ${_fbal.get('cr_total', 0):,.2f} "
                f"(difference ${_fbal.get('difference', 0):,.2f} = {_fbal.get('pct_of_dr', 0):.1f}% of DR). "
                "Explanation required before proceeding."
            )
        elif _bal_flag == "balance_difference_detected":
            st.info(
                f"ℹ️ Small balance difference: ${_fbal.get('difference', 0):,.2f} "
                f"({_fbal.get('pct_of_dr', 0):.2f}%) — likely rounding."
            )

        _column_mapping = _fin.get("column_mapping") or {}
        _row_diag = _fin.get("row_diagnostics") or {}
        _top_rows = _fin.get("top_flagged_rows_preview") or _row_diag.get("top_flagged_rows_preview") or []

        if _column_mapping:
            st.markdown("**Step 5 — Column Mapping**")
            _map_rows = []
            for _canon, _meta in _column_mapping.items():
                _map_rows.append({
                    "Canonical Field": _canon,
                    "Source Column": _meta.get("source_column", ""),
                    "Confidence": _meta.get("confidence", ""),
                })
            st.dataframe(pd.DataFrame(_map_rows), use_container_width=True, hide_index=True)

        if _row_diag:
            st.markdown("**Step 6 — Population Diagnostics**")
            _diag_cols = st.columns(5)
            _diag_cols[0].metric("Rows Retained", int(_row_diag.get("row_count", 0)))
            _diag_cols[1].metric("Flagged Rows", int(_row_diag.get("flagged_row_count", 0)))
            _diag_cols[2].metric("Duplicates", int(_row_diag.get("duplicate_rows", 0)))
            _diag_cols[3].metric("Malformed", int(_row_diag.get("malformed_rows", 0)))
            _diag_cols[4].metric("Outliers", int(_row_diag.get("outlier_rows", 0)))
            _blocking_reasons = _row_diag.get("blocking_reasons") or []
            if _blocking_reasons:
                st.warning("Population blockers: " + "; ".join(_blocking_reasons))
            else:
                st.success("Population diagnostics are within threshold for downstream use.")

            if _top_rows:
                st.markdown("**Top Flagged Rows Preview**")
                st.dataframe(pd.DataFrame(_top_rows), use_container_width=True, hide_index=True)

        # ── Step 7: Confirm / Override ────────────────────────────────────────
        st.markdown("**Review and Confirm**")

        # TB year fast-path
        if _ftype == "trial_balance_unknown_year":
            st.caption("This trial balance needs one more piece of information:")
            _tb1, _tb2, _tb3 = st.columns([1.4, 1.4, 3])
            if _tb1.button("✅ Current Year TB", key=f"tb_current_{selected}", type="primary"):
                doc_specific["_financial"]["doc_type"] = "trial_balance_current"
                doc_specific["_financial"]["finality_state"] = "user_confirmed"
                doc_specific["_financial"]["user_confirmed_type"] = True
                st.session_state["_retry_queue"] = [selected]
                st.session_state["_retry_trigger"] = True
                st.session_state["_retry_locked_type"] = {selected: "trial_balance_current"}
                _save_financial_override(selected, doc_specific["_financial"])
                st.rerun()
            if _tb2.button("📁 Prior Year TB", key=f"tb_prior_{selected}"):
                doc_specific["_financial"]["doc_type"] = "trial_balance_prior_year"
                doc_specific["_financial"]["finality_state"] = "user_confirmed"
                doc_specific["_financial"]["user_confirmed_type"] = True
                st.session_state["_retry_queue"] = [selected]
                st.session_state["_retry_trigger"] = True
                st.session_state["_retry_locked_type"] = {selected: "trial_balance_prior_year"}
                _save_financial_override(selected, doc_specific["_financial"])
                st.rerun()

        # Confirm detected type (for non-TB or any trusted file)
        _conf1, _conf2 = st.columns([2, 3])
        if _ffin != "user_confirmed":
            if _conf1.button(
                f"✅ Confirm as {_TYPE_LABELS.get(_ftype, _ftype)}",
                key=f"confirm_type_{selected}",
                type="primary" if _ffin == "review_required" else "secondary",
            ):
                doc_specific["_financial"]["finality_state"]      = "user_confirmed"
                doc_specific["_financial"]["user_confirmed_type"] = True
                st.session_state["_retry_queue"] = [selected]
                st.session_state["_retry_trigger"] = True
                st.session_state["_retry_locked_type"] = {selected: _ftype}
                _save_financial_override(selected, doc_specific["_financial"])
                st.rerun()

        # Override dropdown (collapsed by default unless review_required)
        with st.expander(
            "🔄 Change document type",
            expanded=(_ffin == "review_required" and _ftype != "trial_balance_unknown_year")
        ):
            _new_type = st.selectbox(
                "Reclassify as:",
                options=_ALL_TYPES,
                format_func=lambda x: _TYPE_LABELS.get(x, x),
                index=_ALL_TYPES.index(_ftype) if _ftype in _ALL_TYPES else 0,
                key=f"reclassify_{selected}",
            )
            if st.button("✅ Confirm Change", key=f"reclassify_confirm_{selected}"):
                doc_specific["_financial"]["doc_type"]            = _new_type
                doc_specific["_financial"]["finality_state"]      = "user_confirmed"
                doc_specific["_financial"]["user_confirmed_type"] = True
                doc_specific["_financial"]["user_override_type"]  = _new_type
                _save_financial_override(selected, doc_specific["_financial"])
                st.session_state["_retry_queue"] = [selected]
                st.session_state["_retry_trigger"] = True
                st.session_state["_retry_locked_type"] = {selected: _new_type}
                st.rerun()

# ── Evidence Readiness Panel ─────────────────────────────────────────────────
_rd = ev.readiness
if _rd:
    _rd_status = _rd.readiness_status
    _rd_colors = {
        "ready":                        ("#f0fdf4", "#16a34a", "✅ Ready to Use"),
        "needs_reviewer_confirmation":  ("#eff6ff", "#1d4ed8", "🔵 Needs Reviewer Confirmation"),
        "needs_client_answer":          ("#fffbeb", "#92400e", "🟡 Needs Client Answer"),
        "exception_open":               ("#fff7ed", "#c2410c", "🟠 Exception Open"),
        "unusable":                     ("#fef2f2", "#991b1b", "❌ Unusable"),
    }
    _bg, _color, _label = _rd_colors.get(_rd_status, ("#f9fafb", "#374151", _rd_status))

    st.markdown(
        f"<div style='background:{_bg};border-left:4px solid {_color};"
        f"padding:10px 16px;border-radius:0 6px 6px 0;margin-bottom:8px'>"
        f"<span style='font-weight:700;color:{_color}'>{_label}</span>"
        + (f" &nbsp;|&nbsp; <span style='color:#6b7280;font-size:0.85rem'>"
           f"{len(_rd.questions)} question(s)</span>"
           if _rd.questions else "")
        + "</div>",
        unsafe_allow_html=True
    )

    _wf = (ev.document_specific or {}).get("_workflow") or {}
    _lineage = _wf.get("lineage") or {}
    if _lineage.get("replaces"):
        st.caption(f"Supersedes prior version: {_lineage['replaces']}")
    if _wf.get("question_history"):
        st.caption(f"Workflow history entries: {len(_wf['question_history'])}")

    # Population readiness for financial files
    if _rd.population_ready is not None:
        if _rd.population_ready:
            st.markdown(
                "<div style='background:#f0fdf4;border:1px solid #86efac;"
                "padding:6px 14px;border-radius:4px;display:inline-block;margin-bottom:8px'>"
                "<span style='color:#15803d;font-weight:600'>📊 Population Ready</span> "
                "<span style='color:#4b5563;font-size:0.82rem'>— rows are clean enough for sampling and analytics</span></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='background:#fff7ed;border:1px solid #fdba74;"
                "padding:6px 14px;border-radius:4px;display:inline-block;margin-bottom:8px'>"
                "<span style='color:#c2410c;font-weight:600'>📊 Evidence Only</span> "
                f"<span style='color:#4b5563;font-size:0.82rem'>— {_rd.population_status}</span></div>",
                unsafe_allow_html=True
            )

    # Questions panel
    if _rd.questions:
        _focused_qid = st.session_state.get("_focus_question_id")
        _reviewer_qs = [q for q in _rd.questions if q.audience == "reviewer"]
        _client_qs   = [q for q in _rd.questions if q.audience == "client"]

        for _qlabel, _qlist, _qcolor in [
            ("🔵 Reviewer Questions", _reviewer_qs, "#1d4ed8"),
            ("🟡 Client Questions",   _client_qs,   "#92400e"),
        ]:
            if not _qlist:
                continue
            st.markdown(f"**{_qlabel}**")
            for _q in _qlist:
                _q_bg = "#f0f9ff" if _q.audience == "reviewer" else "#fffbeb"
                _q_border = "#93c5fd" if _q.audience == "reviewer" else "#fcd34d"
                _block_tag = " <span style='color:#dc2626;font-size:0.75rem;font-weight:600'>[BLOCKING]</span>" if _q.blocking else ""

                _flag_context = _question_detail(ev, _q)
                _bulk_targets = _matching_bulk_targets(selected, _q.question_type, _q.source_flag)

                if _focused_qid == _q.question_id:
                    st.caption("Focused from Questions to Resolve")
                st.markdown(
                    f"<div style='background:{_q_bg};border-left:3px solid {_q_border};"
                    f"padding:10px 14px;margin:4px 0;border-radius:0 4px 4px 0'>"
                    f"<div style='font-size:0.88rem;font-weight:600;margin-bottom:4px'>"
                    f"{_q.question_text}{_block_tag}</div>"
                    + (
                        f"<div style='font-size:0.82rem;color:#374151;background:rgba(0,0,0,0.04);"
                        f"padding:5px 8px;border-radius:3px;margin-top:4px'>"
                        f"<span style='color:#6b7280;font-weight:600'>What was found: </span>"
                        f"{_flag_context}</div>"
                        if _flag_context else ""
                    )
                    + "</div>",
                    unsafe_allow_html=True
                )
                if not _q.resolved:
                    if _bulk_targets:
                        _default_targets = _bulk_targets if len(_bulk_targets) <= 5 else []
                        _apply_files = st.multiselect(
                            "Also apply to matching files",
                            _bulk_targets,
                            default=_default_targets,
                            key=f"detail_apply_{selected}_{_q.question_id}",
                            help="Apply the same answer to other files with the same question type.",
                        )
                    else:
                        _apply_files = []
                    _resolution = st.text_input(
                        "Your answer",
                        key=f"resolution_{selected}_{_q.question_id}",
                        placeholder="Type your response here...",
                    )
                    _rcols = st.columns([2, 1])
                    with _rcols[0]:
                        _resolution_type = st.selectbox(
                            "Resolution type",
                            ["answer", "reviewer_confirmed", "override", "dismissed"],
                            key=f"rtype_{selected}_{_q.question_id}",
                        )
                    with _rcols[1]:
                        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
                        if st.button("✅ Mark resolved", key=f"resolve_{selected}_{_q.question_id}", type="primary"):
                            def _apply_resolution(_ev, _qid=_q.question_id, _res=_resolution, _rtype=_resolution_type):
                                if not _ev.readiness:
                                    _ev.readiness = _rd
                                resolve_question(
                                    _ev, _qid,
                                    _res or "resolved in app",
                                    actor="reviewer",
                                    resolution_type=_rtype,
                                    comment=_res or "resolved in app",
                                )
                            _update_evidence_in_session(selected, _apply_resolution)
                            if _apply_files:
                                _resolve_matching_questions_bulk(_apply_files, _q.question_type, _q.source_flag, _resolution or "resolved in app", _resolution_type)
                            st.rerun()
                else:
                    st.markdown(
                        f"<div style='background:#f0fdf4;border-left:3px solid #86efac;"
                        f"padding:6px 10px;border-radius:3px;font-size:0.82rem;color:#15803d'>"
                        f"✅ Resolved: {_q.resolution or 'resolved'} "
                        f"<span style='color:#6b7280'>({_q.resolution_type or 'answer'})</span></div>",
                        unsafe_allow_html=True
                    )

st.markdown("---")

# ── Section 2: Key Audit Facts ────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Key Audit Facts</div>', unsafe_allow_html=True)

col_l, col_r = st.columns(2)

with col_l:
    if ev.parties:
        st.markdown("**Parties**")
        party_rows = []
        for p in ev.parties:
            prov = p.provenance
            party_rows.append({
                "Role": p.role,
                "Name": p.name,
                "Page": prov.page if prov else "—",
                "Quote": prov.quote if prov else "—",
                "Conf": f"{prov.confidence:.2f}" if prov else "—",
            })
        st.dataframe(pd.DataFrame(party_rows), use_container_width=True, hide_index=True)

    if ev.amounts:
        st.markdown("**Amounts**")
        amt_rows = []
        for a in ev.amounts:
            prov = a.provenance
            amt_rows.append({
                "Type": a.type,
                "Amount": f"${a.value:,.2f}",
                "Page": prov.page if prov else "—",
                "Quote": prov.quote if prov else "—",
                "Conf": f"{prov.confidence:.2f}" if prov else "—",
            })
        st.dataframe(pd.DataFrame(amt_rows), use_container_width=True, hide_index=True)

with col_r:
    if ev.dates:
        st.markdown("**Dates**")
        date_rows = []
        for d in ev.dates:
            prov = d.provenance
            date_rows.append({
                "Type": d.type,
                "Value": d.value,
                "Page": prov.page if prov else "—",
                "Quote": prov.quote if prov else "—",
            })
        st.dataframe(pd.DataFrame(date_rows), use_container_width=True, hide_index=True)

    if ev.identifiers:
        st.markdown("**Identifiers**")
        id_rows = []
        for ident in ev.identifiers:
            prov = ident.provenance
            id_rows.append({
                "Type": ident.type,
                "Value": ident.value,
                "Page": prov.page if prov else "—",
                "Quote": prov.quote if prov else "—",
            })
        st.dataframe(pd.DataFrame(id_rows), use_container_width=True, hide_index=True)

if ev.assets:
    st.markdown("**Assets / Items**")
    asset_rows = [{
        "Type": a.type,
        "Description": a.description,
        "Value": f"${a.value:,.2f}" if a.value else "—",
        "Page": a.provenance.page if a.provenance else "—",
    } for a in ev.assets]
    st.dataframe(pd.DataFrame(asset_rows), use_container_width=True, hide_index=True)

if ev.facts:
    with st.expander(f"📌 Atomic Facts ({len(ev.facts)})", expanded=False):
        fact_rows = [{
            "Label": f.label,
            "Value": str(f.value),
            "Page": f.provenance.page if f.provenance else "—",
            "Quote": f.provenance.quote if f.provenance else "—",
            "Conf": f"{f.provenance.confidence:.2f}" if f.provenance else "—",
        } for f in ev.facts]
        st.dataframe(pd.DataFrame(fact_rows), use_container_width=True, hide_index=True)

# ── Section 3: Claims ─────────────────────────────────────────────────────────
if ev.claims:
    st.markdown('<div class="section-title">📝 Auditor Claims</div>', unsafe_allow_html=True)
    for c in ev.claims:
        prov = c.provenance
        quote_html = f'<br><span class="prov-quote">"{prov.quote}"</span>' if prov and prov.quote else ""
        conf_html = f" | Conf: {prov.confidence:.2f}" if prov else ""
        basis = f" | Based on: {', '.join(c.basis_fact_labels)}" if c.basis_fact_labels else ""
        st.markdown(
            f'<div class="claim-box">'
            f'<strong>{c.statement}</strong>'
            f'<br><small>Area: <strong>{c.audit_area}</strong>'
            f'{conf_html}{basis}'
            f'Page: {prov.page if prov and prov.page else "?"}'
            f'</small>{quote_html}'
            f'</div>',
            unsafe_allow_html=True
        )

# ── Section 4: Flags ──────────────────────────────────────────────────────────
if ev.flags:
    _resolved_flag_types = {q.source_flag for q in (_rd.questions or []) if q.resolved and q.source_flag} if _rd else set()
    _active_flags = [flag for flag in ev.flags if flag.type not in _resolved_flag_types]
    _resolved_flags = [flag for flag in ev.flags if flag.type in _resolved_flag_types]
    if _active_flags or _resolved_flags:
        st.markdown('<div class="section-title">🚩 Flags & Exceptions</div>', unsafe_allow_html=True)
    if _active_flags:
        for flag in _active_flags:
            severity = flag.severity
            st.markdown(
                f'<div class="flag-{severity}">'
                f'<strong>[{severity.upper()}] {flag.type}</strong><br>'
                f'{flag.description}'
                f'</div>',
                unsafe_allow_html=True
            )
    if _resolved_flags:
        with st.expander(f"Resolved exceptions ({len(_resolved_flags)})", expanded=False):
            for flag in _resolved_flags:
                severity = flag.severity
                st.markdown(
                    f'<div class="flag-{severity}">'
                    f'<strong>[{severity.upper()}] {flag.type}</strong><br>'
                    f'{flag.description}'
                    f'</div>',
                    unsafe_allow_html=True
                )

# ── Section 5: Link Keys ──────────────────────────────────────────────────────
lk = ev.link_keys
has_links = any([lk.party_names, lk.document_numbers, lk.invoice_numbers,
                 lk.agreement_numbers, lk.recurring_amounts, lk.other_ids])
if has_links:
    with st.expander("🔗 Link Keys — Cross-Document Matching", expanded=False):
        st.caption("Normalized keys for matching against GL, AP, fixed assets, and other evidence.")
        lk_rows = []
        for field, values in lk.model_dump().items():
            if values:
                lk_rows.append({
                    "Key Type": field.replace("_", " ").title(),
                    "Values": ", ".join(str(v) for v in values)
                })
        if lk_rows:
            st.dataframe(pd.DataFrame(lk_rows), use_container_width=True, hide_index=True)

# ── Section 6: Document Specific ─────────────────────────────────────────────
if ev.document_specific:
    with st.expander("📄 Document-Specific Fields", expanded=False):
        st.json(ev.document_specific)

# ── Section 7: Tables ─────────────────────────────────────────────────────────
if ev.tables:
    with st.expander(f"📊 Extracted Tables ({len(ev.tables)})", expanded=False):
        for i, tbl in enumerate(ev.tables):
            page_n = tbl.get("page_number", tbl.get("page", "?"))
            st.markdown(f"**Table {i+1}** (page {page_n})")
            rows = tbl.get("rows") or []
            if rows:
                try:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                except Exception:
                    st.json(rows[:5])

# ── Extraction Diagnostics ────────────────────────────────────────────────────
with st.expander("🔬 Extraction Diagnostics", expanded=False):
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Extractor", meta.primary_extractor)
    d2.metric("Pages", meta.pages_processed)
    d3.metric("Total Chars", f"{meta.total_chars:,}")
    d4.metric("Confidence", f"{meta.overall_confidence:.2f}")

    d5, d6, d7, d8 = st.columns(4)
    d5.metric("Weak Pages", meta.weak_pages_count)
    d6.metric("OCR Pages", meta.ocr_pages_count)
    d7.metric("Vision Pages", meta.vision_pages_count)
    d8.metric("Needs Review", "Yes" if meta.needs_human_review else "No")

    if meta.warnings:
        st.warning("**Warnings:** " + " | ".join(meta.warnings))
    if meta.errors:
        st.error("**Errors:** " + " | ".join(meta.errors))

    # Engine chain
    chain = r.get("engine_chain", [])
    if chain:
        st.markdown(f"**Engine Chain:** `{' → '.join(chain)}`")

    # Per-stage timing
    doc_specific = ev_data.get("document_specific") or {}
    stage_timings = doc_specific.get("_stage_timings")
    if stage_timings:
        timing_cols = st.columns(len(stage_timings))
        for i, (stage, t) in enumerate(stage_timings.items()):
            timing_cols[i].metric(stage.replace("_", " ").title(), f"{t:.2f}s")


# ── Section 8: Raw Text by Page ───────────────────────────────────────────────
if ev.raw_text:
    with st.expander("📝 Raw Extracted Text", expanded=False):
        st.text(ev.raw_text[:5000])

# ── Section 9: Full Canonical JSON ───────────────────────────────────────────
with st.expander("🔧 Full Canonical JSON", expanded=False):
    st.json(ev.model_dump())
