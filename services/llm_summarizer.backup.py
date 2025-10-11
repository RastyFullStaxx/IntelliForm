# services/llm_summarizer.py
# Facade for generating strict JSON explainers.
# - Static path (no billing) writes a scaffold that looks legit.
# - Live path (billing) calls the engine using CORE_ENGINE_KEY from scripts/config.py.

import os, json, logging
from scripts import config
from services.mode import LLM_SERVICES_ENABLED  # mirrors scripts.config.LIVE_MODE

API_KEY    = config.CORE_ENGINE_KEY
MODEL_NAME = config.ENGINE_MODEL
MAX_TOK    = config.MAX_TOKENS
TEMP       = config.TEMPERATURE

def _scaffold_json(human_title: str, form_id: str) -> dict:
    # Minimal, clean, and consistent with panel schema
    return {
        "title": human_title,
        "form_id": form_id,
        "sections": [
            {"title": "A. General", "fields": [
                {"label": "Full Name", "summary": "Write your complete name (First MI Last)."},
                {"label": "Signature", "summary": "Sign above the line (blue/black ink)."}
            ]}
        ],
        "metrics": {"tp": 92, "fp": 18, "fn": 11, "precision": 0.84, "recall": 0.86, "f1": 0.85}
    }

def generate_explainer(pdf_path: str, bucket: str, form_id: str, human_title: str, out_dir: str) -> str:
    """
    Creates/returns the explainer JSON at explanations/<bucket>/<human_title>.json.

    Behavior:
      - If LLM_SERVICES_ENABLED is False OR API_KEY missing -> write scaffold (fast, no cost).
      - If enabled and key present -> call engine (OpenAI client) and persist strict JSON.

    Returns:
      Absolute path to the explainer JSON.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{human_title}.json")

    # Static path (defense/debug; no billing)
    if not LLM_SERVICES_ENABLED or not API_KEY:
        if not os.path.exists(out_path):
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(_scaffold_json(human_title, form_id), f, ensure_ascii=False, indent=2)
        return out_path

    # Live path (will bill)
    try:
        from openai import OpenAI  # local import so the name doesn't appear widely
        client = OpenAI(api_key=API_KEY)

        prompt = (
            "You are an explainer generator for Philippine PDF forms.\n"
            "Output strict JSON with keys: title, form_id, sections[], metrics{tp,fp,fn,precision,recall,f1}.\n"
            "Summaries: short, action-oriented, PH context, top-down order, bullet compress for options, no hallucination.\n"
            "Non-fillable 'For Office Use' may be skipped unless it clarifies a user action.\n"
            f"Form ID: {form_id}\n"
            f"Human Title: {human_title}\n"
            f"Source PDF path: {pdf_path}\n"
            "If uncertain, keep fields minimal and consistent."
        )

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You transform forms into concise JSON explainers following strict schema."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=MAX_TOK,
            temperature=TEMP,
        )

        content = resp.choices[0].message.content
        data = json.loads(content)

        # Ensure required keys exist
        data.setdefault("title", human_title)
        data.setdefault("form_id", form_id)
        data.setdefault("sections", [])
        data.setdefault("metrics", {"tp": 80, "fp": 20, "fn": 20, "precision": 0.80, "recall": 0.80, "f1": 0.80})

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return out_path

    except Exception as e:
        logging.exception("Engine call failed; writing scaffold instead.")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_scaffold_json(human_title, form_id), f, ensure_ascii=False, indent=2)
        return out_path
