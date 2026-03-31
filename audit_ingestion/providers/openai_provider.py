"""
audit_ingestion_v05/audit_ingestion/providers/openai_provider.py
OpenAI provider — Responses API + Structured Outputs.

Model constants — change here only:
  CANONICAL_MODEL  = "gpt-5.4"      canonical structured extraction
  VISION_MODEL     = "gpt-5.4"      weak-page vision transcription
  RESCUE_MODEL     = "gpt-5.4-pro"  optional manual rescue only
  DEFAULT_MODEL    = CANONICAL_MODEL

PROVIDER BUILD: v05.0
"""
from __future__ import annotations
import base64
import json
import logging
import os
import subprocess
import tempfile
from typing import Optional

from .base import AIProvider

logger = logging.getLogger(__name__)

# ── Model constants — change ONLY here ───────────────────────────────────────
CANONICAL_MODEL = "gpt-5.4"
VISION_MODEL    = "gpt-5.4"
RESCUE_MODEL    = "gpt-5.4-pro"
DEFAULT_MODEL   = CANONICAL_MODEL
PROVIDER_BUILD  = "v05.1"

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class OpenAIProvider(AIProvider):

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        if not HAS_OPENAI:
            raise ImportError("openai not installed. Run: pip install openai")
        self.client = openai.OpenAI(api_key=api_key)
        self.model  = model
        logger.info(f"OpenAI provider ready — model: {model} | build: {PROVIDER_BUILD}")

    # ── Core Responses API call ───────────────────────────────────────────────

    def _responses_call(
        self,
        *,
        system: str,
        user: str,
        model: Optional[str] = None,
        max_output_tokens: int = 4000,
        json_schema: Optional[dict] = None,
    ) -> str:
        """
        Call OpenAI Responses API with explicit content blocks.
        When json_schema provided, uses structured outputs:
          text.format.type   = "json_schema"
          text.format.name   = <schema name>  ← REQUIRED at format level
          text.format.schema = <schema body>
          text.format.strict = True
        Returns raw response text.
        """
        m = model or self.model

        # Explicit content blocks — more robust than bare string content
        input_messages = [
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user",   "content": [{"type": "input_text", "text": user}]},
        ]

        kwargs: dict = dict(
            model=m,
            input=input_messages,
            max_output_tokens=max_output_tokens,
        )

        if json_schema:
            schema_name   = json_schema.get("name", "audit_evidence")
            schema_strict = json_schema.get("strict", True)
            schema_body   = json_schema.get("schema", json_schema)

            fmt = {
                "type":   "json_schema",
                "name":   schema_name,
                "strict": schema_strict,
                "schema": schema_body,
            }
            kwargs["text"] = {"format": fmt}

            # Preflight validation — explicit raises, never bare assert
            if "name" not in fmt:
                raise ValueError("BUG: text.format.name is required by Responses API")
            if fmt.get("type") != "json_schema":
                raise ValueError(f"BUG: unexpected format type: {fmt.get('type')}")
            if "schema" not in fmt:
                raise ValueError("BUG: text.format.schema is required")

            logger.info(
                f"[responses.create] model={m} | "
                f"format.type={fmt['type']} | "
                f"format.name={fmt.get('name', 'MISSING')} | "
                f"format.strict={fmt.get('strict')} | "
                f"format.schema present={('schema' in fmt)} | "
                f"build={PROVIDER_BUILD}"
            )
        else:
            logger.info(
                f"[responses.create] model={m} | "
                f"plain text mode | build={PROVIDER_BUILD}"
            )

        resp = self.client.responses.create(**kwargs)
        return resp.output_text or ""

    # ── Structured canonical extraction ──────────────────────────────────────

    def extract_structured(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict,
        max_tokens: int = 4000,
    ) -> dict:
        """
        Extract structured JSON via Responses API + Structured Outputs.
        Uses CANONICAL_MODEL. Raises ValueError if empty or unparseable.
        """
        logger.info(
            f"extract_structured | model={CANONICAL_MODEL} | "
            f"schema={json_schema.get('name', 'MISSING')} | build={PROVIDER_BUILD}"
        )

        raw = self._responses_call(
            system=system,
            user=user,
            model=CANONICAL_MODEL,
            max_output_tokens=max_tokens,
            json_schema=json_schema,
        )

        if not raw or not raw.strip():
            raise ValueError("Empty response from structured extraction")

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Structured output returned invalid JSON: {e}\nRaw: {raw[:500]}"
            )

    # ── Vision page transcription ─────────────────────────────────────────────

    def extract_text_from_page_images(
        self,
        *,
        images: list[bytes],
        prompt: str,
        model: Optional[str] = None,
    ) -> str:
        """
        Send page images to vision model via Responses API.
        Used for weak-page transcription (VISION_MODEL) and rescue (RESCUE_MODEL).
        Plain text response — no structured output format.
        """
        if not images:
            return ""

        m = model or VISION_MODEL
        logger.info(
            f"extract_text_from_page_images | "
            f"model={m} | images={len(images)} | build={PROVIDER_BUILD}"
        )

        content: list[dict] = []
        for img_bytes in images[:8]:
            b64 = base64.b64encode(img_bytes).decode()
            content.append({
                "type":      "input_image",
                "image_url": f"data:image/jpeg;base64,{b64}",
                "detail":    "high",
            })
        content.append({"type": "input_text", "text": prompt})

        resp = self.client.responses.create(
            model=m,
            input=[{"role": "user", "content": content}],
            max_output_tokens=4000,
        )
        return resp.output_text or ""

    # ── PDF vision fallback ───────────────────────────────────────────────────

    def extract_text_from_pdf_vision(
        self,
        pdf_bytes: bytes,
        max_pages: int = 2,
    ) -> str:
        """Convert PDF pages to images and extract via vision. Last-resort fallback."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        img_dir = tempfile.mkdtemp()
        image_bytes_list: list[bytes] = []

        try:
            subprocess.run(
                ["pdftoppm", "-jpeg", "-r", "150", "-l", str(max_pages),
                 tmp_path, f"{img_dir}/page"],
                capture_output=True, timeout=60,
            )
            for img_file in sorted(os.listdir(img_dir)):
                if img_file.endswith(".jpg"):
                    img_path = os.path.join(img_dir, img_file)
                    with open(img_path, "rb") as f:
                        image_bytes_list.append(f.read())
                    os.unlink(img_path)
                    if len(image_bytes_list) >= max_pages:
                        break
        except Exception as e:
            logger.warning(f"pdftoppm failed: {e}")
        finally:
            try:
                os.unlink(tmp_path)
                os.rmdir(img_dir)
            except Exception:
                pass

        if not image_bytes_list:
            return ""

        return self.extract_text_from_page_images(
            images=image_bytes_list,
            prompt=(
                "Extract ALL text from these document pages faithfully. "
                "Preserve all numbers, dates, names, amounts, and terms exactly as written. "
                "Separate pages with '--- PAGE BREAK ---'."
            ),
        )
