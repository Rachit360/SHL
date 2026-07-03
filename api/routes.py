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
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException

from api.schemas import ChatRequest, ChatResponse, HealthResponse, Recommendation
from rag.prompts import build_catalog_block, build_agent_prompt, CLASSIFIER_SYSTEM, COMPARE_EXTRA
from rag.retriever import get_retriever
from rag.ranking import rank_pool
from rag.catalog_lookup import _load_catalog, _exact_match

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

# ── Behavioral signal detection ─────────────────────────────────────────────

_BEHAVIORAL_PATTERNS = re.compile(
    r"communicat"
    r"|teamwork|team\s+player|team\s+collaboration"
    r"|leadership|leading\s+a\s+team|manage\s+people"
    r"|stakeholder"
    r"|collaborat"
    r"|personality"
    r"|customer\s+(interaction|service|facing)"
    r"|interpersonal"
    r"|people\s+management"
    r"|conflict\s+resolution"
    r"|emotional\s+intelligence"
    r"|adaptability|adaptable"
    r"|negotiat"
    r"|client\s+facing",
    re.IGNORECASE,
)


def _has_behavioral_signal(text: str, constraints: dict) -> bool:
    """Detect whether the job description/query mentions behavioral or
    interpersonal requirements alongside technical ones."""
    if _BEHAVIORAL_PATTERNS.search(text):
        return True
    keywords = constraints.get("keywords") or []
    combined = " ".join(str(k) for k in keywords)
    return bool(_BEHAVIORAL_PATTERNS.search(combined))


def _build_behavioral_query(constraints: dict, text: str) -> str:
    """Build a query string focused specifically on personality/behavioral
    assessments, so retrieval doesn't get crowded out by technical terms."""
    parts = ["personality behavioral communication teamwork assessment"]
    if constraints.get("job_role"):
        parts.append(constraints["job_role"])
    # Pull out only the behavioral phrases actually present, for relevance
    matches = _BEHAVIORAL_PATTERNS.findall(text)
    return " ".join(parts)[:400]


def _merge_pools(primary: list[dict], secondary: list[dict], max_items: int = 15) -> list[dict]:
    """Merge two retrieval pools, deduping by URL, preserving primary-first order."""
    seen = set()
    merged = []
    for a in primary + secondary:
        url = a.get("url", "")
        if url and url not in seen:
            seen.add(url)
            merged.append(a)
        if len(merged) >= max_items:
            break
    return merged


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

def _finalize_recommendations(
    recs: list[Recommendation],
    pool: list[dict],
    state: str,
    max_items: int = 10,
) -> list[Recommendation]:
    """Final guarantee: 1-10 items, no duplicates, ranking order preserved.

    - Dedupes by URL, keeping first occurrence (preserves whatever order
      the LLM/ranking already produced).
    - Truncates to max_items if ranking somehow returned more than 10.
    - For RECOMMEND/REFINE only: if validation left an empty list but the
      retrieval pool had real candidates, backfill from the pool in its
      existing (already-ranked) order rather than returning nothing.
    - CLARIFY/COMPARE/REFUSE are untouched by the backfill — those states
      are allowed, and expected, to have empty recommendations.
    """
    seen_urls = set()
    deduped = []
    for r in recs:
        if r.url not in seen_urls:
            seen_urls.add(r.url)
            deduped.append(r)

    deduped = deduped[:max_items]

    if state in ("RECOMMEND", "REFINE") and not deduped and pool:
        fallback_seen = set()
        for a in pool:
            url = a.get("url", "")
            if not url.startswith("https://www.shl.com") or url in fallback_seen:
                continue
            fallback_seen.add(url)
            deduped.append(Recommendation(
                name=a["name"],
                url=url,
                test_type=a.get("test_type_code", "K"),
            ))
            if len(deduped) >= max_items:
                break
        if deduped:
            logger.warning(
                "[routes] %s returned empty recs despite non-empty pool — "
                "used finalization fallback (%d items)", state, len(deduped)
            )

    return deduped


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


# ── Scope guard — deterministic, runs BEFORE any LLM call ──────────────────

_INJECTION_PATTERNS = re.compile(
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions"
    r"|disregard\s+(the\s+)?(above|previous|prior)"
    r"|forget\s+(your|all|previous)\s+instructions"
    r"|you\s+are\s+now\s+"
    r"|act\s+as\s+(a|an)\s+"
    r"|new\s+instructions?\s*[:=]"
    r"|system\s+prompt"
    r"|developer\s+mode"
    r"|jailbreak"
    r"|reveal\s+your\s+(prompt|instructions)"
    r"|pretend\s+(you|to)\s+"
    r"|override\s+your",
    re.IGNORECASE,
)

_OUT_OF_SCOPE_PATTERNS = re.compile(
    r"\baws\s+certif"
    r"|\bcoursera\b"
    r"|\budemy\b"
    r"|\blinkedin\s+learning\b"
    r"|interview\s+(advice|tips|questions)"
    r"|resume|cv\s+(review|advice|tips)"
    r"|cover\s+letter"
    r"|what\s+salary"
    r"|salary\s+(range|offer|negotiat)"
    r"|legal\s+advice"
    r"|visa\s+(sponsorship|status)"
    r"|weather"
    r"|tell\s+me\s+a\s+joke"
    r"|write\s+(me\s+)?(a\s+)?(poem|song|story)"
    r"|\bpython\s+tutorial\b"
    r"|online\s+course",
    re.IGNORECASE,
)


def _scope_guard(text: str) -> bool:
    """Deterministic pre-check. Returns True if the message should be
    hard-routed to REFUSE without trusting the LLM classifier."""
    if _INJECTION_PATTERNS.search(text):
        return True
    if _OUT_OF_SCOPE_PATTERNS.search(text):
        return True
    return False


# ── Leadership / executive purpose gate ─────────────────────────────────────
# Task: for CXO/director/executive/15+ years hiring scenarios, determine the
# hiring purpose (selection, leadership development, succession planning,
# executive coaching, internal promotion) before recommending. Do not ask if
# the purpose is already known from the conversation.

_LEADERSHIP_TITLE_PATTERNS = re.compile(
    r"\bcxo\b|\bc-suite\b|\bchief\s+\w+\s+officer\b|\bceo\b|\bcfo\b|\bcoo\b|\bcto\b"
    r"|\bdirector\b|\bvp\b|\bvice\s+president\b|\bexecutive\b|\bsenior\s+leadership\b"
    r"|\b1[5-9]\+?\s*years?\b|\b2\d\+?\s*years?\b",
    re.IGNORECASE,
)

_PURPOSE_KEYWORDS = {
    "selection": r"\bselect(ion|ing)?\b|\bhiring\s+(a|for)\b|\bnew\s+hire\b|\bfilling?\s+the\s+role\b",
    "leadership_development": r"\bleadership\s+develop|\bdevelop(ing)?\s+(our|their|his|her)\s+leadership|\bgrow(ing)?\s+leaders?\b",
    "succession_planning": r"\bsuccession\s+plan",
    "executive_coaching": r"\bexecutive\s+coach",
    "internal_promotion": r"\bintern(al)?\s+promot|\bpromot(e|ing)\s+(from\s+within|internally)\b",
}


def _extract_hiring_purpose(text: str) -> str | None:
    for purpose, pattern in _PURPOSE_KEYWORDS.items():
        if re.search(pattern, text, re.IGNORECASE):
            return purpose
    return None


# ── Unavailable-tech gate ────────────────────────────────────────────────────
# Task: if a requested technology (e.g. Rust) isn't in the SHL catalog, ask
# before recommending substitutes instead of silently substituting.

_CATALOG_PATH = Path(__file__).parent.parent / "data" / "assessments.json"
_catalog_vocab_cache: set[str] | None = None


def _get_catalog_vocab() -> set[str]:
    """Build (once, cached) the set of every skill/keyword/name term actually
    present in the catalog, so we can tell whether a requested tech is real."""
    global _catalog_vocab_cache
    if _catalog_vocab_cache is not None:
        return _catalog_vocab_cache
    vocab: set[str] = set()
    try:
        with open(_CATALOG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        for a in data:
            for field in ("skills_measured", "keywords"):
                for term in (a.get(field) or []):
                    vocab.add(str(term).lower().strip())
            vocab.add(str(a.get("name", "")).lower())
    except Exception as e:
        logger.error("[routes] Failed to build catalog vocab: %s", e)
    _catalog_vocab_cache = vocab
    return vocab


# NOTE: curated list of commonly-requested languages/techs that tend NOT to
# have dedicated SHL assessments. Pragmatic, explainable approach rather than
# fully general detection — extend this list as needed.
_TECH_TERM_PATTERN = re.compile(
    r"\b(rust|kotlin|elixir|haskell|erlang|scala|clojure|dart|julia|zig|nim|"
    r"crystal|assembly|fortran|cobol|ada|prolog|lisp|scheme|racket|ocaml|solidity)\b",
    re.IGNORECASE,
)

_ALTERNATIVES_MARKER = "closest alternatives"


def _detect_unavailable_tech(text: str) -> str | None:
    match = _TECH_TERM_PATTERN.search(text)
    if not match:
        return None
    term = match.group(1).lower()
    vocab = _get_catalog_vocab()
    if any(term in v for v in vocab):
        return None  # actually covered by catalog under some entry
    return match.group(1)


def _already_offered_alternatives(messages) -> bool:
    assistant_text = " ".join(m.content for m in messages if m.role == "assistant").lower()
    return _ALTERNATIVES_MARKER in assistant_text


_AFFIRMATIVE_PATTERN = re.compile(
    r"^\s*(yes|yeah|yep|sure|go ahead|please do|ok(ay)?|sounds good|do it)\b",
    re.IGNORECASE,
)


# ── Contact centre / language gate ──────────────────────────────────────────
# Task: for contact centre / customer service hiring, ask for the language
# before recommending, if not already specified.

_CONTACT_CENTRE_PATTERNS = re.compile(
    r"\bcontact\s+cent(re|er)\b|\bcall\s+cent(re|er)\b"
    r"|\bcustomer\s+(service|support)\b|\bcustomer\s+service\s+rep",
    re.IGNORECASE,
)

_LANGUAGE_PATTERNS = re.compile(
    r"\benglish\b|\bspanish\b|\bfrench\b|\bgerman\b|\bhindi\b|\bmandarin\b|\bchinese\b"
    r"|\bportuguese\b|\barabic\b|\bjapanese\b|\bkorean\b|\bitalian\b|\bdutch\b|\brussian\b"
    r"|\btagalog\b|\bfilipino\b|\bvietnamese\b|\bthai\b|\bpolish\b",
    re.IGNORECASE,
)


# ── Confirmation → end_of_conversation ──────────────────────────────────────
# Task: when the user clearly confirms/accepts the final shortlist with no
# further changes requested, acknowledge and set end_of_conversation=true.
# UPDATED (Task A, this round): generalized from an exact-phrase whitelist to
# a pattern match so phrasings like "We'll go with that" and "This is the
# right fit" are also caught, not just the original fixed phrase set.

_CONFIRMATION_PATTERNS = re.compile(
    r"^\s*(confirmed|that works|perfect|sounds good|looks good|all set|"
    r"good to go|we'?ll go with (that|this)|this is the right fit|"
    r"this works|that'?s? all|that covers it|no further changes|"
    r"nothing else( needed)?|great,?\s*thanks|thanks|thank you|"
    r"let'?s go with (that|this)|go with (that|this)|that'?s perfect|"
    r"that'?s great|agreed|approved|yes,?\s*that works|yes,?\s*perfect)"
    r"\s*[.!]*\s*$",
    re.IGNORECASE,
)

_REFINEMENT_SIGNAL = re.compile(
    r"\?|\bbut\b|\balso\b|\bwhat about\b|\bcan you\b|\bcould you\b|\badd\b|"
    r"\bremove\b|\bchange\b|\binstead\b|\bhow about\b",
    re.IGNORECASE,
)


def _is_final_confirmation(text: str) -> bool:
    if _REFINEMENT_SIGNAL.search(text):
        return False
    return bool(_CONFIRMATION_PATTERNS.match(text.strip()))


def _conversation_user_text(messages) -> str:
    return " ".join(m.content for m in messages if m.role == "user")


def _conversation_assistant_text(messages) -> str:
    return " ".join(m.content for m in messages if m.role == "assistant")


# ── Healthcare admin / Spanish language limitation gate (Task B, new) ──────
# Task: when the user requests healthcare admin assessments in Spanish with
# HIPAA requirements, don't immediately recommend. Explain the catalog's
# mixed language availability and ask whether candidates are functionally
# bilingual or need everything in Spanish. Leave recommendations empty
# until the user answers.

_HEALTHCARE_ADMIN_PATTERN = re.compile(
    r"\bhealthcare\s+admin|\bmedical\s+admin|\bhipaa\b|\bhealth\s+administrat",
    re.IGNORECASE,
)
_SPANISH_REQUIREMENT_PATTERN = re.compile(r"\bspanish\b", re.IGNORECASE)
_LANGUAGE_CLARIFIED_PATTERN = re.compile(
    r"\bfunctionally bilingual\b|\ball assessments? in spanish\b|\ball in spanish\b|"
    r"\benglish for written work\b|\bbilingual\b",
    re.IGNORECASE,
)


def _needs_language_limitation_clarification(convo_user_text: str) -> bool:
    if not (_HEALTHCARE_ADMIN_PATTERN.search(convo_user_text)
            and _SPANISH_REQUIREMENT_PATTERN.search(convo_user_text)):
        return False
    if _LANGUAGE_CLARIFIED_PATTERN.search(convo_user_text):
        return False
    return True


# ── Full-stack ambiguity gate (Task D, new) ─────────────────────────────────
# Task: when the user pastes a long job description with many technologies
# spanning both backend and frontend, don't recommend immediately. Ask
# whether the role is backend-focused, frontend-focused, or balanced
# full-stack. Leave recommendations empty until the user answers.

_BACKEND_TECH_PATTERN = re.compile(
    r"\bjava\b|\bspring\b|\brest\b|\bsql\b|\baws\b|\bdocker\b|\bnode\.?js\b|"
    r"\bpython\b|\bmicroservices?\b|\bbackend\b",
    re.IGNORECASE,
)
_FRONTEND_TECH_PATTERN = re.compile(
    r"\bangular\b|\breact\b|\bvue\b|\bcss\b|\bhtml\b|\bjavascript\b|\btypescript\b|"
    r"\bfrontend\b|\bfront-end\b",
    re.IGNORECASE,
)
_ROLE_FOCUS_STATED_PATTERN = re.compile(
    r"\bbackend[- ]focused\b|\bfrontend[- ]focused\b|\bfull[- ]stack\b|"
    r"\bback-end\b|\bfront-end\b|\bbalanced\b|"
    r"\bfocus(ed)? on (backend|frontend|both)\b",
    re.IGNORECASE,
)


def _needs_stack_focus_clarification(last_text: str, convo_user_text: str) -> bool:
    has_backend = bool(_BACKEND_TECH_PATTERN.search(last_text))
    has_frontend = bool(_FRONTEND_TECH_PATTERN.search(last_text))
    if not (has_backend and has_frontend):
        return False
    if len(last_text) < 80:
        return False  # short one-liners aren't "long JD" pastes
    if _ROLE_FOCUS_STATED_PATTERN.search(convo_user_text):
        return False
    return True


# ── Graduate management trainee — pinned battery, no substitution (Task E, new) ──
# Task: when the user requests a graduate management trainee battery covering
# cognitive, personality, and situational judgement, recommend SHL Verify
# Interactive G+, OPQ32r (Occupational Personality Questionnaire OPQ32r in
# the actual catalog), and Graduate Scenarios. Never substitute MQ5 or
# another personality assessment for OPQ32r. Keep end_of_conversation false.

_GRAD_TRAINEE_PATTERN = re.compile(
    r"\bgraduate\s+management\s+trainee\b|"
    r"\bmanagement\s+trainee\b.*\b(cognitive|personality|situational)\b",
    re.IGNORECASE,
)

_GRAD_TRAINEE_TARGET_NAMES = [
    "SHL Verify Interactive G+",
    "Occupational Personality Questionnaire OPQ32r",
    "Graduate Scenarios",
]


def _resolve_named_targets(names: list[str]) -> list[dict]:
    """Best-effort exact-match against the real catalog for a fixed target
    list. Only returns assessments that genuinely exist — never fabricates
    a match if the exact name isn't present."""
    catalog = _load_catalog()
    resolved = []
    for name in names:
        match = _exact_match(name, catalog)
        if match:
            resolved.append(match)
        else:
            logger.warning("[routes] Pinned target not found in catalog: %r", name)
    return resolved


# ── Quick-screen Excel/Word — lightweight tests, mention simulations (Task C, new) ──
# Task: when the user asks to quickly screen admin assistants for Excel and
# Word, recommend the lightweight knowledge tests (MS Excel (New) and MS
# Word (New)), and explicitly mention Microsoft 365 simulation assessments
# are available but not being recommended initially since the user wants a
# quick screen. Keep end_of_conversation as false.

_QUICK_SCREEN_EXCEL_WORD_PATTERN = re.compile(
    r"\bquick(ly)?\s+screen\b.*\b(excel|word)\b|"
    r"\b(excel|word)\b.*\bquick(ly)?\s+screen\b|"
    r"\bscreen\b.*\badmin(istrative)?\s+assistants?\b.*\b(excel|word)\b",
    re.IGNORECASE,
)

_EXCEL_WORD_TARGET_NAMES = ["MS Excel (New)", "MS Word (New)"]


def _find_simulation_assessment(catalog: list[dict]) -> dict | None:
    for a in catalog:
        name = a.get("name", "").lower()
        if "microsoft 365" in name or (
            "simulation" in name and ("excel" in name or "word" in name)
        ):
            return a
    return None


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

     # ── Deterministic scope guard — runs before any LLM call ────────────
    if _scope_guard(last_user):
        logger.warning("[routes] Scope guard triggered, forcing REFUSE: %r", last_user[:120])
        pool: list[dict] = []
        catalog_block = build_catalog_block(pool, max_items=15)
        system_prompt = build_agent_prompt(state="REFUSE", catalog_block=catalog_block)
        try:
            raw = await _call_groq(system_prompt, groq_msgs, temperature=0.2, max_tokens=300)
            parsed = _parse_json(raw)
            reply = parsed.get("reply", "").strip() or (
                "I can only help with SHL assessment selection. "
                "I'm not able to assist with that request."
            )
        except Exception:
            reply = (
                "I can only help with SHL assessment selection. "
                "I'm not able to assist with that request."
            )
        return ChatResponse(
            reply=reply,
            recommendations=[],
            end_of_conversation=False,
        )

    # ── Final confirmation → end conversation ────────────────────────────
    if turn > 1 and _is_final_confirmation(last_user) and _conversation_assistant_text(request.messages).strip():
        constraints = await _extract_constraints(groq_msgs)
        query = _build_query(constraints, last_user)
        confirm_pool = retriever.retrieve(query, top_k=10)
        behavioral_flag = _has_behavioral_signal(last_user, constraints)
        if behavioral_flag:
            behavioral_query = _build_behavioral_query(constraints, last_user)
            behavioral_pool = retriever.retrieve(behavioral_query, top_k=5)
            confirm_pool = _merge_pools(confirm_pool, behavioral_pool, max_items=15)
        confirm_pool = rank_pool(
            confirm_pool, constraints=constraints, last_user_text=last_user,
            behavioral_signal=behavioral_flag, top_k=15,
        )
        recs = _finalize_recommendations([], confirm_pool, "RECOMMEND", max_items=10)
        return ChatResponse(
            reply="Great — glad that works! Here's the finalized shortlist for your hiring process.",
            recommendations=recs,
            end_of_conversation=True,
        )

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

    # ── Deterministic gates before recommending ──────────────────────────
    convo_user_text = _conversation_user_text(request.messages)

    # ── Pinned battery: graduate management trainee (Task E, new) ────────
    if state in ("RECOMMEND", "REFINE") and _GRAD_TRAINEE_PATTERN.search(convo_user_text):
        grad_targets = _resolve_named_targets(_GRAD_TRAINEE_TARGET_NAMES)
        if len(grad_targets) == len(_GRAD_TRAINEE_TARGET_NAMES):
            recs = [
                Recommendation(name=a["name"], url=a["url"], test_type=a.get("test_type_code", "K"))
                for a in grad_targets
            ]
            return ChatResponse(
                reply=(
                    "For a graduate management trainee battery covering cognitive ability, "
                    "personality, and situational judgement, I recommend: SHL Verify Interactive "
                    "G+ (cognitive ability), OPQ32r (personality), and Graduate Scenarios "
                    "(situational judgement). Let me know if you'd like to refine this battery."
                ),
                recommendations=recs,
                end_of_conversation=False,
            )
        # If not all three exact names exist in catalog, fall through to
        # normal RAG flow rather than hallucinating a partial pinned list.

    # ── Pinned battery: quick-screen Excel/Word (Task C, new) ─────────────
    if state == "RECOMMEND" and _QUICK_SCREEN_EXCEL_WORD_PATTERN.search(last_user):
        catalog = _load_catalog()
        excel_word_targets = _resolve_named_targets(_EXCEL_WORD_TARGET_NAMES)
        if excel_word_targets:
            recs = [
                Recommendation(name=a["name"], url=a["url"], test_type=a.get("test_type_code", "K"))
                for a in excel_word_targets
            ]
            sim = _find_simulation_assessment(catalog)
            sim_note = (
                f" Microsoft 365 simulation-based assessments (e.g. {sim['name']}) are also "
                f"available in the catalog, but I'm not including them here since you asked "
                f"for a quick screen — simulations take longer to complete. Let me know if "
                f"you'd like me to add them instead."
                if sim else
                " Microsoft 365 simulation-based assessments are also available in the "
                "catalog, but I'm not including them here since you asked for a quick screen "
                "— simulations take longer to complete. Let me know if you'd like me to add "
                "them instead."
            )
            return ChatResponse(
                reply=(
                    "For a quick screen, I recommend MS Excel (New) and MS Word (New) — "
                    "these are lightweight knowledge tests." + sim_note
                ),
                recommendations=recs,
                end_of_conversation=False,
            )
        # Falls through to normal flow if exact names aren't found —
        # no hallucination risk, just misses this specific shortcut.

    # ── Gate: healthcare admin + Spanish + mixed language availability (Task B, new) ──
    if state == "RECOMMEND" and _needs_language_limitation_clarification(convo_user_text):
        return ChatResponse(
            reply=(
                "The SHL catalog has mixed language availability for this role: HIPAA, "
                "Medical Terminology, and Microsoft Word assessments are English-only, "
                "while personality assessments like DSI and OPQ32r support Spanish. Are "
                "the candidates functionally bilingual (comfortable with English for "
                "written assessments), or do you need every assessment delivered in Spanish?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    # ── Gate: backend/frontend/full-stack ambiguity (Task D, new) ─────────
    if state == "RECOMMEND" and _needs_stack_focus_clarification(last_user, convo_user_text):
        return ChatResponse(
            reply=(
                "This job description includes both backend and frontend technologies. "
                "Is this role primarily backend-focused, frontend-focused, or a balanced "
                "full-stack position? This will help me tailor the assessment battery."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    # Gate: unavailable tech (e.g. Rust) — ask before substituting
    if state == "RECOMMEND":
        tech = _detect_unavailable_tech(last_user)
        if tech and not _already_offered_alternatives(request.messages):
            alt_pool = retriever.retrieve(f"{tech} alternative programming assessment", top_k=5)
            alt_names = [a["name"] for a in alt_pool[:3]] or ["Linux Programming", "Networking", "Live Coding"]
            reply = (
                f"There isn't a {tech.capitalize()}-specific assessment in the SHL catalog. "
                f"Would you like recommendations for the closest alternatives instead, "
                f"such as {', '.join(alt_names)}?"
            )
            return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

    # Fallback: classifier said CLARIFY on the "yes, go ahead" follow-up turn
    if state == "CLARIFY" and _already_offered_alternatives(request.messages) and _AFFIRMATIVE_PATTERN.match(last_user):
        state = "RECOMMEND"

    # Gate: leadership/executive purpose unknown
    if state == "RECOMMEND":
        if _LEADERSHIP_TITLE_PATTERNS.search(convo_user_text) and not _extract_hiring_purpose(convo_user_text):
            reply = (
                "Before I recommend leadership assessments, could you tell me the purpose "
                "of this evaluation? For example: selection, leadership development, "
                "succession planning, executive coaching, or internal promotion."
            )
            return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

    # Gate: contact centre language unknown
    if state == "RECOMMEND":
        if _CONTACT_CENTRE_PATTERNS.search(convo_user_text) and not _LANGUAGE_PATTERNS.search(convo_user_text):
            return ChatResponse(
                reply="What language will the calls be handled in?",
                recommendations=[],
                end_of_conversation=False,
            )

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

            # ── Behavioral signal boost ──────────────────────────────────
            if _has_behavioral_signal(last_user, constraints):
                behavioral_query = _build_behavioral_query(constraints, last_user)
                behavioral_pool  = retriever.retrieve(behavioral_query, top_k=5)
                pool = _merge_pools(pool, behavioral_pool, max_items=15)
                logger.info("[routes] Behavioral signal detected, merged pool size=%d", len(pool))

   
        
         # ── Weighted multi-factor ranking (skip for COMPARE) ─────────────
        if state != "COMPARE":
            behavioral_flag = _has_behavioral_signal(last_user, constraints)
            pool = rank_pool(
                pool,
                constraints=constraints,
                last_user_text=last_user,
                behavioral_signal=behavioral_flag,
                top_k=15,
            )
            logger.info(
                "[routes] Ranked pool top score=%.3f | size=%d",
                pool[0]["_rank_score"] if pool else 0.0, len(pool),
            )

    # ── Build prompt and call LLM ────────────────────────────────────────
    catalog_block = build_catalog_block(pool, max_items=15)
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
    valid_recs = _finalize_recommendations(valid_recs, pool, state)

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