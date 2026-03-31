"""
audit_ingestion_v05/audit_ingestion/router.py
Pipeline orchestrator with v05 segmentation step.

Stage 1: Extract           → ParsedDocument (page-aware, v4)
Stage 2: Segment (NEW)     → SegmentationResult (primary + attachments)
Stage 3: Canonical AI      → AuditEvidence (primary pages only)
Stage 4: Normalize         → normalized evidence
Stage 5: Score + annotate  → IngestionResult with segmentation info
"""
from __future__ import annotations
import logging
import threading
import time
from pathlib import Path
from typing import Optional
from .models import (
    AuditEvidence, IngestionResult, ExtractionMeta, Flag,
    SegmentationResult, DocumentComponent, AttachmentSummary, DocumentFamily,
)
from .extractor import extract
from .normalizers import normalize_evidence
from .financial_classifier import classify_financial_file, is_financial_file, TYPE_NOT_FINANCIAL
from .readiness import apply_readiness

logger = logging.getLogger(__name__)

_AI_SEMAPHORE  = threading.Semaphore(2)
_OCR_SEMAPHORE = threading.Semaphore(2)

ROUTER_BUILD = "v05.0"


def ingest_one(
    path: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    mode: str = "fast",
    allow_rescue: bool = False,
    financial_type_override: Optional[str] = None,
    financial_sheet_override: Optional[str] = None,
    bypass_cache: bool = False,
) -> IngestionResult:
    """
    Ingest one document through the v05 pipeline.
    Segmentation runs automatically — the user never configures it.
    financial_type_override: if set, skips Layer 1/2 classification and
    uses this type directly (user_confirmed finality). Used after reclassification.
    """
    input_p = Path(path)
    engine_chain: list[str] = []
    errors: list[str] = []
    stage_timings: dict[str, float] = {}

    if not input_p.exists():
        return IngestionResult(
            status="failed",
            errors=["File not found"],
            evidence=AuditEvidence(
                source_file=input_p.name,
                flags=[Flag(type="file_not_found", description="File not found", severity="critical")]
            ),
        )

    # Provider init
    provider = None
    if api_key:
        try:
            from .providers import get_provider
            provider = get_provider("openai", api_key=api_key, model=model)
        except Exception as e:
            errors.append(f"Provider init failed: {e}")
            logger.error(f"Provider init: {e}")
    else:
        errors.append("No API key — extraction only, no canonical analysis")

    # Stage 1: Extract
    t0 = time.perf_counter()
    try:
        parsed_doc = extract(path, provider=provider, mode=mode,
                             ocr_semaphore=_OCR_SEMAPHORE)
        stage_timings["extraction"] = round(time.perf_counter() - t0, 3)
        engine_chain.extend(parsed_doc.extraction_chain)
        if parsed_doc.errors:
            errors.extend(parsed_doc.errors)
    except Exception as e:
        stage_timings["extraction"] = round(time.perf_counter() - t0, 3)
        errors.append(f"Extraction failed: {e}")
        return IngestionResult(
            status="failed", errors=errors,
            evidence=AuditEvidence(
                source_file=input_p.name,
                flags=[Flag(type="extraction_error", description=str(e), severity="critical")]
            ),
        )

    meta = ExtractionMeta(
        primary_extractor=parsed_doc.primary_extractor,
        pages_processed=parsed_doc.page_count,
        weak_pages_count=len(parsed_doc.weak_pages),
        ocr_pages_count=len(parsed_doc.ocr_pages),
        vision_pages_count=len(parsed_doc.vision_pages),
        total_chars=len(parsed_doc.full_text),
        overall_confidence=parsed_doc.confidence,
        needs_human_review=not parsed_doc.is_sufficient,
        warnings=parsed_doc.warnings,
        errors=errors,
    )

    # Minimal evidence object — always available for early returns
    # Gets replaced by canonical AI extraction if pipeline proceeds
    evidence = AuditEvidence(
        source_file=input_p.name,
        raw_text=parsed_doc.full_text,
        tables=[t if isinstance(t, dict) else t.model_dump() for t in parsed_doc.tables],
        extraction_meta=meta,
    )

    # Stage 1b: Financial classification (CSV/Excel files)
    financial_data: Optional[dict] = None
    if is_financial_file(path):
        t_fin = time.perf_counter()
        try:
            financial_data = classify_financial_file(
                path,
                provider=provider,
                type_override=financial_type_override,
                sheet_override=financial_sheet_override,
            )
            stage_timings["financial_classification"] = round(time.perf_counter() - t_fin, 3)
            engine_chain.append("financial_classified")
            logger.info(
                f"Financial: {input_p.name} → "
                f"{financial_data.get('doc_type','?')} "
                f"(finality={financial_data.get('finality_state','?')})"
            )
        except Exception as e:
            stage_timings["financial_classification"] = round(time.perf_counter() - t_fin, 3)
            logger.warning(f"Financial classification failed: {e}")
            financial_data = None

    # Early gate: stop here if financial classification requires user confirmation.
    # Do not run segmentation or canonical AI — both waste credits and are
    # meaningless until the user confirms the document type.
    if financial_data and financial_data.get("finality_state") == "review_required":
        logger.info(
            f"review_required — stopping pipeline for {input_p.name} "
            f"(doc_type={financial_data.get('doc_type','?')}). "
            f"Awaiting user confirmation before canonical extraction."
        )
        _annotate_with_financial_data(evidence, financial_data)

        # Set family/subtype from financial classification so the summary
        # table shows the correct type instead of "other"
        _fin_doc_type = financial_data.get("doc_type", "")
        _fin_type_to_family = {
            "general_ledger":             ("accounting_report", "general_ledger"),
            "journal_entry_listing":      ("accounting_report", "journal_entry_listing"),
            "trial_balance_unknown_year": ("accounting_report", "trial_balance"),
            "trial_balance_current":      ("accounting_report", "trial_balance_current"),
            "trial_balance_prior_year":   ("accounting_report", "trial_balance_prior_year"),
            "budget":                     ("accounting_report", "budget"),
            "bank_statement_csv":         ("bank_cash_activity", "bank_statement_csv"),
            "chart_of_accounts":          ("accounting_report", "chart_of_accounts"),
        }
        if _fin_doc_type in _fin_type_to_family:
            _fam_str, _sub_str = _fin_type_to_family[_fin_doc_type]
            try:
                evidence.family = DocumentFamily(_fam_str)
            except ValueError:
                pass
            evidence.subtype = _sub_str

        apply_readiness(evidence)
        score = _score(evidence)
        evidence.extraction_meta.overall_confidence = score
        stage_timings["segmentation"]  = 0.0
        stage_timings["canonical_ai"]  = 0.0
        stage_timings["total"] = round(sum(
            v for k, v in stage_timings.items()
            if isinstance(v, float) and k != "total"
        ), 3)
        evidence.document_specific["_stage_timings"] = stage_timings
        return IngestionResult(
            source_file=path,
            status="partial",
            evidence=evidence,
            errors=errors,
            engine_chain=engine_chain,
        )

    # Stage 2: Segment (NEW in v05)
    # Also skipped for financial structured files — CSVs/Excel have no page bundles
    segmentation: Optional[SegmentationResult] = None
    extraction_doc = parsed_doc  # what gets passed to canonical AI

    if provider is not None and parsed_doc.full_text and len(parsed_doc.pages) > 2             and not financial_data:  # financial files never need bundle segmentation
        t1 = time.perf_counter()
        try:
            from .segmenter import segment, build_primary_document
            segmentation = segment(parsed_doc, provider)
            stage_timings["segmentation"] = round(time.perf_counter() - t1, 3)
            engine_chain.append("segmented")

            if segmentation.bundle_detected:
                # Scope canonical extraction to primary component pages only
                extraction_doc = build_primary_document(parsed_doc, segmentation)
                logger.info(
                    f"Bundle detected: {parsed_doc.source_file} | "
                    f"primary={segmentation.primary_page_count} pages | "
                    f"attachments={len(segmentation.attachment_components)}"
                )
            else:
                logger.info(f"No bundle: {parsed_doc.source_file} — single document")

        except Exception as e:
            stage_timings["segmentation"] = round(time.perf_counter() - t1, 3)
            logger.warning(f"Segmentation failed: {e} — proceeding with full document")
            segmentation = None
    else:
        stage_timings["segmentation"] = 0.0

    # Attach financial data to extraction doc so canonical AI can use it
    if financial_data and hasattr(extraction_doc, '__dict__'):
        # Fix: use setattr directly — __fields__ check was deprecated Pydantic v1 pattern
        try:
            setattr(extraction_doc, '_financial_data', financial_data)
        except Exception:
            pass  # ParsedDocument is a dataclass — setattr always works

    if provider is not None and extraction_doc.full_text:
        t2 = time.perf_counter()
        with _AI_SEMAPHORE:
            try:
                from .canonical import extract_canonical
                evidence = extract_canonical(extraction_doc, provider, mode=mode, bypass_cache=bypass_cache)
                engine_chain.append("canonical_ai")
            except Exception as e:
                errors.append(f"Canonical extraction failed: {e}")
                logger.error(f"Canonical: {e}")
                evidence = AuditEvidence(
                    source_file=input_p.name,
                    raw_text=extraction_doc.full_text,
                    tables=[t if isinstance(t, dict) else t.model_dump()
                            for t in extraction_doc.tables],
                    extraction_meta=meta,
                    flags=[Flag(type="canonical_failed", description=str(e), severity="critical")]
                )
                engine_chain.append("canonical_failed")
        stage_timings["canonical_ai"] = round(time.perf_counter() - t2, 3)

        # Stage 3b: Rescue (allow_rescue path — unchanged from v04)
        if allow_rescue and parsed_doc.weak_pages and provider is not None:
            worst_pages = sorted(
                [p for p in parsed_doc.pages if p.page_number in parsed_doc.weak_pages],
                key=lambda p: p.char_count
            )[:2]
            if worst_pages:
                t_rescue = time.perf_counter()
                try:
                    from .providers.openai_provider import RESCUE_MODEL
                    from .extractor import render_page_image_cached
                    rescue_texts = []
                    with _AI_SEMAPHORE:
                        for pg in worst_pages:
                            img = render_page_image_cached(path, pg.page_number - 1, dpi=200)
                            if not img:
                                continue
                            rescued = provider.extract_text_from_page_images(
                                images=[img],
                                prompt=f"Read page {pg.page_number} of this document image. "
                                       f"Extract all audit-relevant facts. Return plain text only.",
                                model=RESCUE_MODEL,
                            )
                            if rescued and rescued.strip():
                                rescue_texts.append(f"[Rescued page {pg.page_number}]\n{rescued.strip()}")
                                engine_chain.append(f"rescue_p{pg.page_number}")
                    if rescue_texts:
                        evidence.flags.append(Flag(
                            type="rescue_applied",
                            description=f"gpt-5.4-pro visual rescue applied to {len(rescue_texts)} page(s).",
                            severity="info",
                        ))
                        evidence.document_specific["rescued_page_text"] = "\n\n".join(rescue_texts)
                    stage_timings["rescue"] = round(time.perf_counter() - t_rescue, 3)
                except Exception as e:
                    logger.warning(f"Rescue failed: {e}")
                    stage_timings["rescue"] = 0.0

    else:
        flag_type = "no_ai" if not api_key else "no_text"
        flag_desc = "No API key — canonical extraction skipped" if not api_key else "No text extracted"
        evidence = AuditEvidence(
            source_file=input_p.name,
            raw_text=parsed_doc.full_text,
            tables=[t if isinstance(t, dict) else t.model_dump() for t in parsed_doc.tables],
            extraction_meta=meta,
            flags=[Flag(type=flag_type, description=flag_desc, severity="warning")]
        )
        engine_chain.append("extraction_only")
        stage_timings["canonical_ai"] = 0.0

    # Stage 4: Normalize
    t3 = time.perf_counter()
    try:
        evidence = normalize_evidence(evidence)
        engine_chain.append("normalized")
    except Exception as e:
        logger.warning(f"Normalization failed: {e}")
    stage_timings["normalization"] = round(time.perf_counter() - t3, 3)

    # For financial files: remove partial_extraction flags — these files are
    # extracted deterministically so "partial extraction" is not meaningful.
    # The canonical AI fires this flag when it sees structured data and assumes
    # it couldn't read everything, but we already have all the data via the
    # financial classifier.
    if financial_data and financial_data.get("doc_type") != TYPE_NOT_FINANCIAL:
        evidence.flags = [
            f for f in evidence.flags
            if f.type not in (
                "partial_extraction", "partial_extraction_visibility",
                "incomplete_extraction", "truncated_content",
            )
        ]

    # Stage 5: Annotate with segmentation info + score
    _annotate_with_segmentation(evidence, segmentation)
    if financial_data:
        _annotate_with_financial_data(evidence, financial_data)



    # Compute evidence readiness and generate questions
    apply_readiness(evidence)

    score = _score(evidence)
    ai_unavailable = any(f.type in ("canonical_failed", "no_ai") for f in evidence.flags)
    has_text = (evidence.extraction_meta.total_chars or 0) >= 200
    if ai_unavailable and has_text and score < 0.30:
        score = 0.30

    evidence.extraction_meta.overall_confidence = score
    # Deduplicate flags — same type can appear multiple times from cache replay
    seen_flag_types: set[str] = set()
    deduped_flags = []
    for _flag in evidence.flags:
        if _flag.type not in seen_flag_types:
            seen_flag_types.add(_flag.type)
            deduped_flags.append(_flag)
    evidence.flags = deduped_flags

    evidence.extraction_meta.needs_human_review = (
        score < 0.70 or
        (evidence.readiness and evidence.readiness.blocking_state == "blocking")
    )
    evidence.document_specific["_stage_timings"] = stage_timings
    stage_timings["total"] = round(sum(stage_timings.values()), 3)

    status = "success" if score >= 0.70 else ("partial" if score >= 0.30 else "failed")

    return IngestionResult(
        evidence=evidence,
        status=status,
        errors=errors,
        engine_chain=engine_chain,
    )


def _annotate_with_segmentation(
    evidence: AuditEvidence,
    segmentation: Optional[SegmentationResult],
) -> None:
    """Add segmentation info to the evidence record. Modifies in place."""
    if segmentation is None:
        return

    if segmentation.bundle_detected and segmentation.has_attachments:
        # User-facing flag — clean language, no technical internals
        attachment_names = ", ".join(a.name for a in segmentation.attachment_components)
        evidence.flags.append(Flag(
            type="bundle_detected",
            description=(
                f"Primary document identified: {segmentation.primary_component.description}. "
                f"Supporting attachments separated: {attachment_names}. "
                f"Core facts extracted from primary document only."
            ),
            severity="info",
        ))

        # Store attachment summaries in document_specific for UI display
        evidence.document_specific["_segmentation"] = {
            "bundle_detected":        True,
            "confidence_band":        segmentation.confidence_band,
            "primary_description":    segmentation.primary_component.description,
            "primary_pages":          segmentation.primary_component.pages,
            "attachments": [
                {
                    "name":            a.name,
                    "pages":           a.pages,
                    "summary":         a.summary,
                    "key_identifiers": a.key_identifiers,
                }
                for a in segmentation.attachment_components
            ],
        }

    if segmentation.conservative_note:
        evidence.flags.append(Flag(
            type="conservative_extraction",
            description=segmentation.conservative_note,
            severity="info",
        ))


def _annotate_with_financial_data(
    evidence: AuditEvidence,
    financial_data: dict,
) -> None:
    """Store financial classification results in document_specific. Modifies in place."""
    doc_type    = financial_data.get("doc_type", TYPE_NOT_FINANCIAL)
    finality    = financial_data.get("finality_state", "")
    confidence  = financial_data.get("doc_type_confidence", 0.0)

    # Store full financial data in document_specific
    evidence.document_specific["_financial"] = financial_data

    # Surface balance issues as flags
    bal = financial_data.get("balance_check", {})
    flag_level = bal.get("flag_level", "")
    if flag_level == "material_balance_difference":
        diff = bal.get("difference", 0)
        pct  = bal.get("pct_of_dr", 0)
        evidence.flags.append(Flag(
            type="material_balance_difference",
            description=(
                f"Trial balance is materially out of balance: "
                f"DR ${bal.get('dr_total',0):,.2f} vs CR ${bal.get('cr_total',0):,.2f} "
                f"(difference ${diff:,.2f} = {pct:.1f}% of DR total). "
                f"Explanation required before audit proceeds."
            ),
            severity="warning",
        ))
    elif flag_level == "balance_difference_detected":
        evidence.flags.append(Flag(
            type="balance_difference_detected",
            description=(
                f"Trial balance has a small difference: ${bal.get('difference',0):,.2f}. "
                f"Likely rounding or timing — confirm before relying on totals."
            ),
            severity="info",
        ))

    # Flag missing period for financial files
    if doc_type != TYPE_NOT_FINANCIAL and not financial_data.get("period_start"):
        evidence.flags.append(Flag(
            type="missing_period",
            description=(
                f"Could not determine fiscal period for {doc_type}. "
                f"Check filename or file header for year/period information."
            ),
            severity="info",
        ))

    # Flag TB that needs year confirmation
    if doc_type == "trial_balance_unknown_year":
        evidence.flags.append(Flag(
            type="tb_year_unconfirmed",
            description=(
                "Trial balance year not resolved. "
                "Confirm whether this is the current year or prior year TB."
            ),
            severity="warning",
        ))


def _score(ev: AuditEvidence) -> float:
    """
    Score the quality of a canonical evidence record.
    Financial files (CSV/Excel) use a separate scoring path because they
    structurally lack parties/provenance/dates — those fields are N/A, not missing.
    """
    ds = ev.document_specific or {}
    fin = ds.get("_financial", {})
    is_financial = bool(fin and fin.get("doc_type") and
                        fin["doc_type"] != "not_financial_structured_data")

    if is_financial:
        return _score_financial(ev, fin)
    return _score_document(ev)


def _score_financial(ev: AuditEvidence, fin: dict) -> float:
    """
    Score a pre-classified financial file.
    Rewards: correct classification, period detection, totals, balance check,
             canonical AI claims, no critical errors.
    Financial files do not lose points for missing parties/provenance —
    those fields are not applicable to structured data.
    """
    s = 0.0

    # 1. Classification quality (0-0.25)
    conf = float(fin.get("doc_type_confidence", 0))
    finality = fin.get("finality_state", "")
    if finality in ("trusted", "user_confirmed"):
        s += 0.25 * conf
    elif finality == "review_recommended":
        s += 0.20 * conf
    else:  # review_required
        s += 0.10 * conf

    # 2. Period detected (0-0.15)
    period_conf = float(fin.get("period_confidence", 0))
    if fin.get("period_start"):
        if period_conf >= 0.90:
            s += 0.15
        elif period_conf >= 0.65:
            s += 0.10
        else:
            s += 0.05

    # 3. Totals extracted (0-0.20)
    totals = fin.get("totals", {})
    non_error_totals = {k: v for k, v in totals.items() if not k.endswith("_error")}
    if len(non_error_totals) >= 4:
        s += 0.20
    elif len(non_error_totals) >= 2:
        s += 0.12
    elif len(non_error_totals) >= 1:
        s += 0.06

    # 4. Balance check result (0-0.15)
    bal = fin.get("balance_check", {})
    bal_flag = bal.get("flag_level", "")
    if bal_flag == "tb_balanced":
        s += 0.15   # clean balance is a positive signal
    elif bal_flag == "balance_difference_detected":
        s += 0.10   # small diff — still extracted correctly
    elif bal_flag == "material_balance_difference":
        s += 0.10   # still extracted correctly, just flagged
    elif not bal_flag and fin.get("doc_type") not in (
        "trial_balance_unknown_year", "trial_balance_current", "trial_balance_prior_year"
    ):
        s += 0.10   # non-TB types don't need a balance check

    # 5. Canonical AI quality (0-0.15)
    if ev.audit_overview and ev.audit_overview.summary:
        s += 0.05
    if ev.amounts:
        s += 0.05
    if ev.claims:
        s += min(0.05, len(ev.claims) * 0.02)

    # 6. No critical errors (0-0.10)
    critical_flags = [f for f in (ev.flags or []) if f.severity == "critical"]
    if not critical_flags:
        s += 0.10

    return round(min(s, 1.0), 3)


def _score_document(ev: AuditEvidence) -> float:
    """Score a standard document (PDF, text). Original scoring logic."""
    s = 0.0
    if ev.audit_overview and ev.audit_overview.summary:
        s += 0.20
    s += 0.07 if ev.amounts  else 0
    s += 0.07 if ev.parties  else 0
    s += 0.06 if ev.dates    else 0
    s += 0.05 if ev.facts    else 0
    if ev.claims:
        s += min(0.15, len(ev.claims) * 0.05)
    all_items = (
        [(a.provenance, a.value) for a in ev.amounts] +
        [(p.provenance, p.name) for p in ev.parties] +
        [(d.provenance, d.value) for d in ev.dates]
    )
    if all_items:
        with_prov = sum(1 for prov, _ in all_items if prov and prov.confidence > 0.5)
        s += 0.20 * (with_prov / len(all_items))
    lk = ev.link_keys
    if any([lk.party_names, lk.document_numbers, lk.invoice_numbers,
            lk.agreement_numbers, lk.recurring_amounts]):
        s += 0.10
    if ev.extraction_meta.total_chars >= 500:
        s += 0.10
    elif ev.extraction_meta.total_chars >= 200:
        s += 0.05
    return round(min(s, 1.0), 3)
