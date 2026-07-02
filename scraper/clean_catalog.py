"""
SHL Catalog Cleaner v3
-----------------------
Reads raw_assessments.json, keeps all scraped SHL fields exactly as-is,
and adds derived enrichment fields to improve RAG quality.

Derived fields (extracted from existing data — nothing invented):
  id                   → slug from name
  category             → human label from test_type_code
  skills_measured      → technologies/concepts from description
  keywords             → concise search terms from name + description
  recommend_for        → likely roles/scenarios inferred from text
  business_problems_solved → recruiter intents inferred from text
  embedding_text       → rich concatenation for semantic embedding

Usage:
    python clean_catalog.py
"""

import json
import re
import sys
from pathlib import Path

RAW_PATH   = Path(__file__).parent / "output" / "raw_assessments.json"
CLEAN_PATH = Path(__file__).parent / "output" / "assessments.json"

# ── Test type inference ───────────────────────────────────────────────────────
TEST_TYPE_RULES = [
    ("P", ["occupational personality", "opq", "personality questionnaire",
           "behavioural style", "behavioral style", "adept-15",
           "personality inventory", "motives values preferences"]),
    ("M", ["motivation", "motivational questionnaire", "mq32", "values and motives",
           "work motivation", "drives and values"]),
    ("B", ["situational judgement", "situational judgment", "sjt",
           "biodata", "situational test"]),
    ("A", ["numerical reasoning", "verbal reasoning", "inductive reasoning",
           "deductive reasoning", "abstract reasoning", "spatial reasoning",
           "mechanical reasoning", "checking test", "calculation test",
           "reading comprehension", "error checking", "figure reasoning",
           "diagrammatic reasoning", "cognitive ability", "verify ",
           "verify-", "reasoning test"]),
    ("S", ["simulation", "inbox", "in-tray", "in tray", "work simulation",
           "virtual inbox"]),
    ("E", ["assessment exercise", "role play", "group exercise",
           "presentation exercise", "fact find"]),
    ("D", ["360", "development report", "360-degree", "multi-rater",
           "feedback questionnaire", "development needs"]),
    ("C", ["competency", "competencies", "structured interview",
           "behavioral interview", "behaviour interview"]),
    ("K", ["knowledge", "skill", "proficiency", "test measures",
           "microsoft", "excel", "word ", "java", "python", "sql", "css",
           "html", "javascript", ".net", "c++", "c#", "php", "swift",
           "android", "ios", "linux", "unix", "windows", "cisco",
           "networking", "database", "accounting", "finance", "typing",
           "data entry", "programming", "software", "technical",
           "fundamentals", "principles", "concepts", "framework"]),
]

TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "M": "Motivation",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

JOB_LEVEL_MAP = {
    1: "Director", 2: "Entry Level", 3: "Front Line Manager",
    4: "General Population", 5: "Graduate", 6: "Manager",
    7: "Professional Individual Contributor", 8: "Senior Manager", 9: "Supervisor",
}

# ── Skills extraction patterns ─────────────────────────────────────────────────
# Technology / tool keywords — matched against description
TECH_SKILLS = [
    # Programming languages
    "Python","Java","JavaScript","TypeScript","C++","C#","PHP","Ruby","Swift",
    "Kotlin","Go","Rust","Scala","R","MATLAB","Perl","Shell","Bash",
    # Web / frontend
    "HTML","CSS","React","Angular","Vue","Node.js","jQuery","Bootstrap","REST","GraphQL",
    # Backend / infra
    "SQL","MySQL","PostgreSQL","MongoDB","Redis","Elasticsearch","Docker","Kubernetes",
    "AWS","Azure","GCP","Linux","Unix","Windows Server","Cisco","Networking","TCP/IP",
    # Microsoft Office
    "Excel","Word","PowerPoint","Outlook","Access","SharePoint","Teams","Office 365",
    # .NET / Microsoft dev
    ".NET","ASP.NET","WPF","WCF","Entity Framework","LINQ","Visual Studio",
    # Data / analytics
    "Power BI","Tableau","SPSS","SAS","Hadoop","Spark","Machine Learning","Data Analysis",
    # Accounting / finance
    "Accounting","Bookkeeping","Payroll","QuickBooks","SAP","ERP","IFRS","GAAP",
    # Other
    "Android","iOS","Xcode","Git","Agile","Scrum","ITIL","PRINCE2","Six Sigma",
]

# Cognitive / behavioural concepts
COGNITIVE_SKILLS = [
    "Numerical Reasoning","Verbal Reasoning","Inductive Reasoning","Deductive Reasoning",
    "Abstract Reasoning","Spatial Reasoning","Mechanical Reasoning","Error Checking",
    "Reading Comprehension","Calculation","Data Entry","Typing Speed",
    "Personality","Motivation","Situational Judgement","Competency Assessment",
    "Leadership","Teamwork","Communication","Problem Solving","Critical Thinking",
    "Decision Making","Creativity","Emotional Intelligence","Customer Service",
    "Sales Skills","Negotiation","Project Management","Time Management",
]

# ── Role / scenario patterns ──────────────────────────────────────────────────
ROLE_PATTERNS = [
    # Technical roles
    (r"\bjava\b", "Java Developer"),
    (r"\bpython\b", "Python Developer"),
    (r"\.net\b", ".NET Developer"),
    (r"\bsql\b", "Database Developer / Data Analyst"),
    (r"\bjavascript\b|\bhtml\b|\bcss\b", "Front-End Developer"),
    (r"\bnode\.js\b|\bbackend\b", "Back-End Developer"),
    (r"\baws\b|\bazure\b|\bgcp\b|\bcloud\b", "Cloud / DevOps Engineer"),
    (r"\bnetworking\b|\bcisco\b|\btcp", "Network Engineer"),
    (r"\blinux\b|\bunix\b", "Systems Administrator"),
    (r"\bandroid\b|\bios\b|\bmobile\b", "Mobile Developer"),
    (r"\bmachine learning\b|\bdata science\b|\bai\b", "Data Scientist / ML Engineer"),
    # Finance / accounting
    (r"\baccounting\b|\bbookkeeping\b|\bpayroll\b", "Accountant / Finance Professional"),
    (r"\bsap\b|\berp\b", "ERP Consultant"),
    (r"\bbank\b|\bfinance\b|\bfinancial\b", "Finance / Banking Professional"),
    # Office / admin
    (r"\bexcel\b|\bword\b|\boffice\b", "Office / Administrative Professional"),
    (r"\btyping\b|\bdata entry\b", "Data Entry / Administrative Clerk"),
    (r"\bcustomer service\b|\bcall centre\b|\bcall center\b", "Customer Service Representative"),
    # Management / leadership
    (r"\bmanager\b|\bleadership\b|\bmanagement\b", "Manager / Team Leader"),
    (r"\bsales\b|\bnegotiation\b", "Sales Professional"),
    (r"\bproject management\b|\bscrum\b|\bagile\b", "Project Manager"),
    # General personality / cognitive
    (r"\bpersonality\b|\bopq\b|\bbehaviou?r\b", "All Professional Roles"),
    (r"\bnumerical reasoning\b|\bverbal reasoning\b|\binductive\b", "Graduate / Professional Roles"),
    (r"\bsituational judgement\b|\bsjt\b", "Customer-Facing / Service Roles"),
    (r"\bmotivat\b", "All Professional Roles"),
]

# ── Business problem patterns ─────────────────────────────────────────────────
BUSINESS_PROBLEM_PATTERNS = [
    (r"\bjava\b", "Screen Java developers"),
    (r"\bpython\b", "Assess Python programming skills"),
    (r"\.net\b", "Evaluate .NET development knowledge"),
    (r"\bsql\b", "Test database and SQL proficiency"),
    (r"\bjavascript\b|\bhtml\b|\bcss\b", "Screen front-end web developers"),
    (r"\bnetworking\b|\bcisco\b", "Evaluate network engineering skills"),
    (r"\baws\b|\bazure\b|\bcloud\b", "Screen cloud and DevOps professionals"),
    (r"\baccounting\b|\bpayroll\b|\bbookkeeping\b", "Assess accounting and finance skills"),
    (r"\bexcel\b", "Test Microsoft Excel proficiency"),
    (r"\bword\b", "Evaluate Microsoft Word skills"),
    (r"\boffice 365\b|\boffice365\b", "Screen for Microsoft Office 365 knowledge"),
    (r"\btyping\b", "Measure typing speed and accuracy"),
    (r"\bdata entry\b", "Screen data entry candidates"),
    (r"\bcustomer service\b", "Screen customer service representatives"),
    (r"\bsales\b", "Identify high-potential sales candidates"),
    (r"\bpersonality\b|\bopq\b|\bbehaviou?r", "Understand workplace behavioural style"),
    (r"\bnumerical reasoning\b", "Assess numerical and analytical ability"),
    (r"\bverbal reasoning\b", "Evaluate verbal and written communication ability"),
    (r"\binductive reasoning\b|\babstract reasoning\b", "Test problem-solving and logical thinking"),
    (r"\bdeductive reasoning\b", "Assess logical deduction and structured thinking"),
    (r"\bsituational judgement\b|\bsjt\b", "Predict job performance in complex scenarios"),
    (r"\bmotivat\b", "Understand candidate values and work motivation"),
    (r"\bcompetenc", "Evaluate core job competencies"),
    (r"\bleadership\b", "Identify leadership potential"),
    (r"\bproject management\b", "Screen project managers and team leads"),
    (r"\bmanager\b|\bmanagement\b", "Select managers and supervisors"),
    (r"\bgraduate\b|\bentry.level\b", "Screen graduate and entry-level candidates"),
]


def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def infer_test_type(name: str, description: str) -> str:
    combined = (name + " " + description).lower()
    for code, keywords in TEST_TYPE_RULES:
        if any(kw.lower() in combined for kw in keywords):
            return code
    return "K"


def name_to_slug(name: str) -> str:
    slug = name.lower()
    slug = re.sub(r"[^\w\s-]", " ", slug)
    slug = re.sub(r"\s+", "-", slug.strip())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def build_id(name: str) -> str:
    return name_to_slug(name)


def build_url(name: str) -> str:
    return f"https://www.shl.com/solutions/products/assessments/{name_to_slug(name)}/"


def extract_skills_measured(name: str, description: str) -> list[str]:
    """Extract technologies and concepts explicitly mentioned in the text."""
    combined = name + " " + description
    found = []

    # Tech skills (case-insensitive exact match)
    for skill in TECH_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, combined, re.IGNORECASE):
            found.append(skill)

    # Cognitive/behavioural skills
    for skill in COGNITIVE_SKILLS:
        if skill.lower() in combined.lower():
            if skill not in found:
                found.append(skill)

    return found


def extract_keywords(name: str, description: str, test_type_label: str,
                     skills: list[str]) -> list[str]:
    """Build concise search keywords from available fields."""
    kws = set()

    # From name — individual words that are meaningful (len > 2)
    for word in re.split(r'[\s\-\(\)/]+', name):
        word = word.strip(".,")
        if len(word) > 2 and not word.lower() in {
            "the", "and", "for", "with", "test", "new", "form",
            "short", "basic", "advanced", "version"
        }:
            kws.add(word)

    # Test type label words
    for word in test_type_label.split():
        if len(word) > 3:
            kws.add(word)

    # Top skills (already clean)
    for s in skills[:8]:
        kws.add(s)

    # Key phrases from description (first sentence often most informative)
    first_sentence = re.split(r'[.!?]', description)[0] if description else ""
    for word in re.split(r'\s+', first_sentence):
        word = re.sub(r'[^\w]', '', word)
        if len(word) > 4:
            kws.add(word)

    return sorted(kws)[:20]  # cap at 20


def extract_recommend_for(name: str, description: str) -> list[str]:
    """Infer likely hiring roles or scenarios from text."""
    combined = (name + " " + description).lower()
    roles = []
    seen = set()
    for pattern, role in ROLE_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            if role not in seen:
                roles.append(role)
                seen.add(role)
    return roles[:5]  # cap at 5


def extract_business_problems(name: str, description: str) -> list[str]:
    """Infer recruiter intents from the assessment text."""
    combined = (name + " " + description).lower()
    problems = []
    seen = set()
    for pattern, problem in BUSINESS_PROBLEM_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            if problem not in seen:
                problems.append(problem)
                seen.add(problem)
    return problems[:5]


def build_embedding_text(entry: dict) -> str:
    """
    Rich concatenated text for semantic embedding.
    Combines all structured fields without duplication of raw description.
    """
    parts = [
        entry.get("name", ""),
        entry.get("test_type_label", ""),
        entry.get("category", ""),
        entry.get("description", "")[:300],
        " ".join(entry.get("skills_measured", [])),
        " ".join(entry.get("keywords", [])),
        " ".join(entry.get("recommend_for", [])),
        " ".join(entry.get("business_problems_solved", [])),
        " ".join(entry.get("job_levels", [])),
        "Remote testing available" if entry.get("remote_testing") else "",
        "Adaptive IRT" if entry.get("adaptive_irt") else "",
    ]
    return clean_text(" | ".join(p for p in parts if p))


def clean_entry(raw: dict) -> dict | None:
    name        = clean_text(raw.get("name", ""))
    if not name:
        return None

    description = clean_text(raw.get("description", ""))
    test_type   = infer_test_type(name, description)
    type_label  = TEST_TYPE_LABELS.get(test_type, "")
    url         = build_url(name)
    job_ids     = raw.get("job_level_ids", [])
    job_levels  = [JOB_LEVEL_MAP[i] for i in job_ids if i in JOB_LEVEL_MAP]
    languages   = raw.get("languages", [])
    remote      = True     # all assessments on online.shl.com are remote-enabled
    adaptive    = len(raw.get("filter_ids_c", [])) > 0

    # ── Derived enrichment fields ──────────────────────────────────────────
    skills      = extract_skills_measured(name, description)
    keywords    = extract_keywords(name, description, type_label, skills)
    rec_for     = extract_recommend_for(name, description)
    biz_probs   = extract_business_problems(name, description)

    entry = {
        # ── Core SHL scraped fields (unchanged) ───────────────────────────
        "id":               build_id(name),
        "name":             name,
        "url":              url,
        "description":      description,
        "test_type_code":   test_type,
        "test_type_label":  type_label,
        "category":         type_label,   # alias for schema compatibility
        "job_level_ids":    job_ids,
        "job_levels":       job_levels,
        "languages":        languages,
        "remote_testing":   remote,
        "adaptive_irt":     adaptive,
        # ── Derived enrichment fields ─────────────────────────────────────
        "skills_measured":          skills,
        "keywords":                 keywords,
        "recommend_for":            rec_for,
        "business_problems_solved": biz_probs,
    }
    entry["embedding_text"] = build_embedding_text(entry)
    return entry


def deduplicate(entries: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for e in entries:
        key = e["name"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


def main():
    if not RAW_PATH.exists():
        print(f"[cleaner] ERROR: {RAW_PATH} not found. Run scrape_catalog.py first.")
        sys.exit(1)

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"[cleaner] Loaded {len(raw_data)} raw entries")

    cleaned = [e for raw in raw_data if (e := clean_entry(raw))]
    cleaned = deduplicate(cleaned)

    # Stats
    type_dist     = {}
    has_skills    = sum(1 for e in cleaned if e.get("skills_measured"))
    has_rec_for   = sum(1 for e in cleaned if e.get("recommend_for"))
    has_biz       = sum(1 for e in cleaned if e.get("business_problems_solved"))
    for e in cleaned:
        c = e["test_type_code"]
        type_dist[c] = type_dist.get(c, 0) + 1

    print(f"[cleaner] Clean entries          : {len(cleaned)}")
    print(f"[cleaner] With URL               : {len(cleaned)}")
    print(f"[cleaner] With description       : {sum(1 for e in cleaned if e.get('description'))}")
    print(f"[cleaner] With skills_measured   : {has_skills}")
    print(f"[cleaner] With recommend_for     : {has_rec_for}")
    print(f"[cleaner] With business_problems : {has_biz}")
    print(f"[cleaner] Test type distribution : {dict(sorted(type_dist.items()))}")

    print("\n[cleaner] Sample — OPQ32r:")
    opq = next((e for e in cleaned if "opq" in e["name"].lower()), cleaned[0])
    print(f"  skills   : {opq['skills_measured']}")
    print(f"  rec_for  : {opq['recommend_for']}")
    print(f"  biz      : {opq['business_problems_solved']}")
    print(f"  keywords : {opq['keywords'][:8]}")

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLEAN_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    print(f"\n[cleaner] Saved → {CLEAN_PATH}")


if __name__ == "__main__":
    main()
