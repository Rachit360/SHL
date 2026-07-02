"""
enrich_catalog.py
-----------------
Takes final_assessments.json (317 entries with real URLs/duration)
and regenerates all enrichment fields from the official descriptions.

Adds: skills_measured, keywords, recommend_for, business_problems_solved, embedding_text

Usage:
    python scraper/enrich_catalog.py
"""

import json, re, sys
from pathlib import Path

INPUT_PATH  = Path(__file__).parent / "output" / "final_assessments.json"
OUTPUT_PATH = Path(__file__).parent / "output" / "final_assessments.json"
DATA_OUTPUT = Path(__file__).parent.parent / "data" / "assessments.json"

TECH_SKILLS = [
    "Python","Java","JavaScript","TypeScript","C++","C#","PHP","Ruby","Swift",
    "Kotlin","Go","Rust","Scala","R","MATLAB","Perl","Shell","Bash",
    "HTML","CSS","React","Angular","Vue","Node.js","jQuery","REST","GraphQL",
    "SQL","MySQL","PostgreSQL","MongoDB","Redis","Docker","Kubernetes",
    "AWS","Azure","GCP","Linux","Unix","Cisco","TCP/IP",
    "Excel","Word","PowerPoint","Outlook","SharePoint","Office 365",
    ".NET","ASP.NET","WPF","WCF","LINQ","ADO.NET","MVC","MVVM","XAML",
    "Power BI","Tableau","SPSS","Hadoop","Spark",
    "Accounting","Bookkeeping","Payroll","SAP","ERP",
    "Android","iOS","Git","Agile","Scrum","ITIL","Six Sigma",
    "Verify","OPQ","MQ","SJT","ADEPT",
]

COGNITIVE_SKILLS = [
    "Numerical Reasoning","Verbal Reasoning","Inductive Reasoning",
    "Deductive Reasoning","Abstract Reasoning","Spatial Reasoning",
    "Mechanical Reasoning","Error Checking","Reading Comprehension",
    "Situational Judgement","Personality","Motivation","Competency",
    "Leadership","Teamwork","Communication","Problem Solving",
    "Critical Thinking","Decision Making","Customer Service","Sales Skills",
]

ROLE_PATTERNS = [
    (r"\bjava\b",                               "Java Developer"),
    (r"\bpython\b",                             "Python Developer"),
    (r"\.net\b",                                ".NET Developer"),
    (r"\bsql\b",                                "Database Developer"),
    (r"\bjavascript\b|\bhtml\b|\bcss\b",        "Front-End Developer"),
    (r"\bnode\.js\b",                           "Back-End Developer"),
    (r"\baws\b|\bazure\b|\bgcp\b|\bcloud\b",    "Cloud / DevOps Engineer"),
    (r"\bnetworking\b|\bcisco\b",               "Network Engineer"),
    (r"\blinux\b|\bunix\b",                     "Systems Administrator"),
    (r"\bandroid\b|\bios\b",                    "Mobile Developer"),
    (r"\bexcel\b|\bword\b|\boffice\b",          "Office / Administrative Professional"),
    (r"\btyping\b|\bdata entry\b",              "Data Entry / Administrative Clerk"),
    (r"\bcustomer service\b",                   "Customer Service Representative"),
    (r"\bsales\b",                              "Sales Professional"),
    (r"\bmanager\b|\bleadership\b",             "Manager / Team Leader"),
    (r"\bproject management\b",                 "Project Manager"),
    (r"\baccounting\b|\bpayroll\b",             "Accountant / Finance Professional"),
    (r"\bpersonality\b|\bbehaviou?r\b|\bopq\b", "All Professional Roles"),
    (r"\bnumerical reasoning\b",                "Graduate / Analyst Roles"),
    (r"\bverbal reasoning\b",                   "Graduate / Professional Roles"),
    (r"\binductive\b|\babstract reasoning\b",   "Graduate / Professional Roles"),
    (r"\bsituational judgement\b|\bsjt\b",      "Customer-Facing / Service Roles"),
    (r"\bmotivat\b",                            "All Professional Roles"),
    (r"\bsimulation\b|\binbox\b",               "Manager / Professional Roles"),
]

BIZ_PATTERNS = [
    (r"\bjava\b",                           "Screen Java developers"),
    (r"\bpython\b",                         "Assess Python programming skills"),
    (r"\.net\b",                            "Evaluate .NET development knowledge"),
    (r"\bsql\b",                            "Test database and SQL proficiency"),
    (r"\bjavascript\b|\bhtml\b|\bcss\b",    "Screen front-end web developers"),
    (r"\baws\b|\bazure\b|\bcloud\b",        "Screen cloud and DevOps professionals"),
    (r"\baccounting\b|\bpayroll\b",         "Assess accounting and finance skills"),
    (r"\bexcel\b",                          "Test Microsoft Excel proficiency"),
    (r"\btyping\b",                         "Measure typing speed and accuracy"),
    (r"\bdata entry\b",                     "Screen data entry candidates"),
    (r"\bcustomer service\b",               "Screen customer service representatives"),
    (r"\bsales\b",                          "Identify high-potential sales candidates"),
    (r"\bpersonality\b|\bopq\b|\bbehaviou?r","Understand workplace behavioural style"),
    (r"\bnumerical reasoning\b",            "Assess numerical and analytical ability"),
    (r"\bverbal reasoning\b",               "Evaluate verbal communication ability"),
    (r"\binductive\b|\babstract\b",         "Test problem-solving and logical thinking"),
    (r"\bsituational judgement\b|\bsjt\b",  "Predict performance in complex scenarios"),
    (r"\bmotivat\b",                        "Understand candidate values and motivation"),
    (r"\bleadership\b",                     "Identify leadership potential"),
    (r"\bsimulation\b",                     "Assess performance in realistic job scenarios"),
    (r"\bcompetenc\b",                      "Evaluate core job competencies"),
    (r"\bgraduate\b",                       "Screen graduate and entry-level candidates"),
    (r"\bmanager\b|\bmanagement\b",         "Select managers and supervisors"),
]


def extract_skills(name, desc):
    combined = name + " " + desc
    found = []
    for skill in TECH_SKILLS:
        if re.search(r'\b' + re.escape(skill) + r'\b', combined, re.IGNORECASE):
            found.append(skill)
    for skill in COGNITIVE_SKILLS:
        if skill.lower() in combined.lower() and skill not in found:
            found.append(skill)
    return found


def extract_keywords(name, desc, type_label, skills):
    kws = set()
    for word in re.split(r'[\s\-\(\)/]+', name):
        word = word.strip(".,")
        if len(word) > 2 and word.lower() not in {
            "the","and","for","with","test","new","form",
            "short","basic","advanced","version","new"
        }:
            kws.add(word)
    for word in type_label.split():
        if len(word) > 3:
            kws.add(word)
    for s in skills[:6]:
        kws.add(s)
    return sorted(kws)[:20]


def extract_recommend_for(name, desc):
    combined = (name + " " + desc).lower()
    roles, seen = [], set()
    for pattern, role in ROLE_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE) and role not in seen:
            roles.append(role)
            seen.add(role)
    return roles[:5]


def extract_biz_problems(name, desc):
    combined = (name + " " + desc).lower()
    problems, seen = [], set()
    for pattern, problem in BIZ_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE) and problem not in seen:
            problems.append(problem)
            seen.add(problem)
    return problems[:5]


def build_embedding_text(entry):
    parts = [
        entry.get("name", ""),
        entry.get("test_type_label", ""),
        entry.get("description", "")[:300],
        " ".join(entry.get("skills_measured", [])),
        " ".join(entry.get("keywords", [])),
        " ".join(entry.get("recommend_for", [])),
        " ".join(entry.get("business_problems_solved", [])),
        " ".join(entry.get("job_levels", [])),
        f"Duration: {entry.get('duration', {}).get('display', '')}" if isinstance(entry.get('duration'), dict) and entry['duration'].get('display') else "",
        "Remote testing available" if entry.get("remote_testing") else "",
        "Adaptive IRT" if entry.get("adaptive_irt") else "",
    ]
    return " | ".join(p for p in parts if p)


def main():
    if not INPUT_PATH.exists():
        print(f"[enrich] ERROR: {INPUT_PATH} not found.")
        sys.exit(1)

    with open(INPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    print(f"[enrich] Processing {len(data)} assessments ...")

    for entry in data:
        name = entry.get("name", "")
        desc = entry.get("description", "")
        label = entry.get("test_type_label", "")

        skills   = extract_skills(name, desc)
        keywords = extract_keywords(name, desc, label, skills)
        rec_for  = extract_recommend_for(name, desc)
        biz      = extract_biz_problems(name, desc)

        entry["skills_measured"]          = skills
        entry["keywords"]                 = keywords
        entry["recommend_for"]            = rec_for
        entry["business_problems_solved"] = biz
        entry["embedding_text"]           = build_embedding_text(entry)

    # Stats
    has_skills = sum(1 for e in data if e.get("skills_measured"))
    has_rec    = sum(1 for e in data if e.get("recommend_for"))
    has_biz    = sum(1 for e in data if e.get("business_problems_solved"))

    print(f"[enrich] With skills_measured   : {has_skills}/{len(data)}")
    print(f"[enrich] With recommend_for     : {has_rec}/{len(data)}")
    print(f"[enrich] With business_problems : {has_biz}/{len(data)}")

    print(f"\n[enrich] Sample:")
    for e in data[:2]:
        print(f"  {e['name']}")
        print(f"    skills  : {e['skills_measured'][:4]}")
        print(f"    rec_for : {e['recommend_for'][:2]}")
        print(f"    biz     : {e['business_problems_solved'][:2]}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n[enrich] Saved → {OUTPUT_PATH}")

    with open(DATA_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[enrich] Saved → {DATA_OUTPUT}")


if __name__ == "__main__":
    main()
