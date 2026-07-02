"""
Routes
------
POST /chat  — stateless conversational agent
GET  /health — readiness probe
"""

import json
import logging
import os
import re
import time

import httpx
from fastapi import APIRouter, HTTPException

from api.schemas import ChatRequest, ChatResponse, HealthResponse, Recommendation
from rag.prompts import build_catalog_block, build_agent_prompt, CLASSIFIER_SYSTEM, COMPARE_EXTRA
from rag.retriever import get_retriever

logger = logging.getLogger(__name__)
router = APIRouter()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE    = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL    = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TIMEOUT  = 25
MAX_TURNS    = 8


# ── Helpers ────────────────────────────────────────────────────────────────

def _messages_to_groq(messages) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _last_user_message(messages) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


async def _call_groq(system: str, messages: list[dict],
                     temperature: float = 0.2, max_tokens: int = 800) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    payload = {
        "model":           LLM_MODEL,
        "messages":        [{"role": "system", "content": system}] + messages,
        "temperature":     temperature,
        "max_tokens":      max_tokens,
        "response_format": {"type": "json_object"},
    }
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        resp = await client.post(
            GROQ_BASE,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                     "Content-Type": "application/json"},
            json=payload,
        )
    if resp.status_code != 200:
        logger.error("Groq error %d: %s", resp.status_code, resp.text[:300])
        raise HTTPException(status_code=502, detail="LLM provider error")
    return resp.json()["choices"][0]["message"]["content"]


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _validate_recommendations(raw_recs: list, pool: list[dict]) -> list[Recommendation]:
    """Only accept assessments that exist in the retrieval pool. Drop hallucinations."""
    name_map = {a["name"].lower(): a for a in pool}
    valid    = []

    for item in raw_recs:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        url  = str(item.get("url", "")).strip()
        code = str(item.get("test_type", "")).strip().upper()

        # Find in catalog by name
        catalog = name_map.get(name.lower())
        if not catalog and url:
            catalog = next((a for a in pool if a.get("url") == url), None)
        if not catalog:
            logger.warning("[routes] Dropping hallucinated rec: %r", name)
            continue

        safe_url = catalog.get("url", "")
        if not safe_url.startswith("https://www.shl.com"):
            logger.warning("[routes] Dropping bad URL: %r", safe_url)
            continue

        # Use catalog test type code if LLM got it wrong
        cat_code = catalog.get("test_type_code", "K")
        if code not in "ABCDEKMPQS":
            code = cat_code

        try:
            valid.append(Recommendation(
                name=catalog["name"],
                url=safe_url,
                test_type=code,
            ))
        except Exception as e:
            logger.warning("[routes] Invalid rec skipped: %s", e)

        if len(valid) >= 10:
            break

    return valid


def _fallback_response(turn: int) -> ChatResponse:
    return ChatResponse(
        reply="I had trouble generating a response. Could you describe the role you're hiring for?",
        recommendations=[],
        end_of_conversation=turn >= MAX_TURNS - 1,
    )


# ── State classifier ───────────────────────────────────────────────────────

async def _classify(messages: list[dict]) -> str:
    try:
        raw    = await _call_groq(CLASSIFIER_SYSTEM, messages, temperature=0.0, max_tokens=80)
        parsed = _parse_json(raw)
        state  = parsed.get("state", "CLARIFY").upper()
        if state not in ("CLARIFY", "RECOMMEND", "REFINE", "COMPARE", "REFUSE"):
            state = "CLARIFY"
        logger.info("[routes] State: %s | %s", state, parsed.get("reasoning", ""))
        return state
    except Exception as e:
        logger.error("[routes] Classifier failed: %s", e)
        return "CLARIFY"


# ── Constraint extraction ──────────────────────────────────────────────────

CONSTRAINT_SYSTEM = """\
Extract hiring constraints from the conversation.
Output ONLY valid JSON with these keys (omit if not mentioned):
{
  "job_role": "string or null",
  "seniority": "entry|mid|senior|executive|null",
  "test_type_codes": ["K","P",...],
  "keywords": ["string", ...]
}
Type codes: A=Ability, B=Biodata/SJT, C=Competencies, D=Development,
E=Exercises, K=Knowledge, M=Motivation, P=Personality, S=Simulations.
Output JSON only. No explanation.
"""


async def _extract_constraints(messages: list[dict]) -> dict:
    try:
        raw = await _call_groq(CONSTRAINT_SYSTEM, messages, temperature=0.0, max_tokens=150)
        return _parse_json(raw)
    except Exception:
        return {}


def _build_query(constraints: dict, last_msg: str) -> str:
    parts = []
    if constraints.get("job_role"):
        parts.append(constraints["job_role"])
    if constraints.get("seniority"):
        parts.append(f"{constraints['seniority']} level")
    if constraints.get("keywords"):
        parts.extend(constraints["keywords"][:4])
    parts.append(last_msg)
    return " ".join(parts)[:400]


def _extract_named_assessments(text: str) -> list[str]:
    m = re.search(r"between\s+(.+?)\s+and\s+(.+?)[\?\.!]", text, re.IGNORECASE)
    if m:
        return [m.group(1).strip(), m.group(2).strip()]
    quoted = re.findall(r'["\']([^"\']+)["\']', text)
    if quoted:
        return quoted
    acronyms = re.findall(
        r"\b(OPQ32r?|MQ|ADEPT-15|Verify[^\s]*|RemoteWorkQ|"
        r"Numerical|Verbal|Inductive|Deductive|SJT|CCSQ)\b",
        text, re.IGNORECASE,
    )
    return list(set(acronyms))


# ── Routes ─────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    t0        = time.monotonic()
    turn      = len(request.messages)
    retriever = get_retriever()
    groq_msgs = _messages_to_groq(request.messages)
    last_user = _last_user_message(request.messages)

    # ── Hard turn cap ────────────────────────────────────────────────────
    if turn >= MAX_TURNS:
        constraints = await _extract_constraints(groq_msgs)
        query = _build_query(constraints, last_user)
        pool  = retriever.retrieve(query, top_k=5)
        recs  = [
            Recommendation(
                name=a["name"], url=a["url"],
                test_type=a.get("test_type_code", "K"),
            )
            for a in pool if a.get("url", "").startswith("https://www.shl.com")
        ][:5]
        return ChatResponse(
            reply="We've reached the conversation limit. Here are my best recommendations:",
            recommendations=recs,
            end_of_conversation=True,
        )

    # ── Classify state ───────────────────────────────────────────────────
    state = await _classify(groq_msgs)

    # ── Retrieve relevant assessments ────────────────────────────────────
    pool: list[dict] = []
    if state in ("RECOMMEND", "REFINE", "COMPARE"):
        constraints = await _extract_constraints(groq_msgs)
        query = _build_query(constraints, last_user)

        if state == "COMPARE":
            named = _extract_named_assessments(last_user)
            pool  = retriever.retrieve_for_comparison(named) if named else []
            if not pool:
                pool = retriever.retrieve(query, top_k=10)
        else:
            pool = retriever.retrieve(query, top_k=10)

    logger.info("[routes] State=%s | Pool=%d | %.2fs",
                state, len(pool), time.monotonic() - t0)

    # ── Build prompt and call LLM ────────────────────────────────────────
    catalog_block = build_catalog_block(pool, max_items=10)
    system_prompt = build_agent_prompt(state=state, catalog_block=catalog_block)
    if state == "COMPARE":
        system_prompt += f"\n\n{COMPARE_EXTRA}"

    if time.monotonic() - t0 > LLM_TIMEOUT - 5:
        return _fallback_response(turn)

    try:
        raw = await _call_groq(system_prompt, groq_msgs, temperature=0.3, max_tokens=600)
    except Exception as e:
        logger.error("[routes] LLM call failed: %s", e)
        return _fallback_response(turn)

    parsed = _parse_json(raw)
    if not parsed:
        return _fallback_response(turn)

    # ── Validate recommendations ─────────────────────────────────────────
    raw_recs = parsed.get("recommendations", [])
    if state in ("CLARIFY", "REFUSE"):
        raw_recs = []

    valid_recs = _validate_recommendations(raw_recs, pool)

    reply   = parsed.get("reply", "").strip() or "Could you tell me more about the role?"
    end_eoc = bool(parsed.get("end_of_conversation", False))
    if turn >= MAX_TURNS - 2 and valid_recs:
        end_eoc = True

    logger.info("[routes] Done. State=%s Recs=%d EOC=%s | %.2fs",
                state, len(valid_recs), end_eoc, time.monotonic() - t0)

    return ChatResponse(
        reply=reply,
        recommendations=valid_recs,
        end_of_conversation=end_eoc,
    )
