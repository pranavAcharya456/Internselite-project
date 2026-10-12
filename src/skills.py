import re
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocess import clean_text

SKILLS = {
    "python": [], "java": [], "javascript": ["js"], "typescript": [], "c++": [], "c#": [],
    "c programming": [], "r programming": ["rstudio"], "sql": [], "html": [], "css": [], "php": [],
    "ruby on rails": ["ruby programming"], "scala": [], "matlab": [], "bash": ["shell scripting"], "linux": [], "git": ["github"],
    "node.js": ["nodejs"], "react.js": ["reactjs", "react js", "react native"], "angularjs": ["angular.js", "angular js", "angular framework"], "django": [], "flask": [],
    "spring boot": ["spring framework", "spring mvc", "spring security"], ".net": ["dotnet", "asp.net"], "rest api": ["restful", "rest apis"],
    "json": [], "xml": [], "mysql": [], "postgresql": ["postgres"], "mongodb": [], "oracle database": ["oracle db", "oracle sql", "pl/sql", "oracle 11g", "oracle 12c"],
    "hadoop": [], "spark": ["pyspark"], "pandas": [], "numpy": [], "scikit-learn": ["sklearn"],
    "tensorflow": [], "pytorch": [], "machine learning": ["ml model", "ml models", "ml algorithms"], "deep learning": [],
    "nlp": ["natural language processing"], "data analysis": ["data analytics"],
    "data visualization": [], "statistics": ["statistical analysis"], "etl": [],
    "tableau": [], "power bi": ["powerbi"],
    "microsoft excel": ["ms excel", "advanced excel", "vlookup", "pivot table", "pivot tables"],
    "microsoft office": ["ms office", "microsoft word", "powerpoint"],
    "aws": ["amazon web services"], "azure": [], "gcp": ["google cloud"], "docker": [],
    "kubernetes": [], "jenkins": [], "ci/cd": ["continuous integration"], "devops": [],
    "networking": [], "tcp/ip": [], "cisco": [], "vmware": [], "active directory": [],
    "cybersecurity": ["information security", "network security"], "help desk": ["helpdesk"],
    "troubleshooting": [], "sap": [], "salesforce": [], "jira": [], "agile": [], "scrum": [],
    "accounting": [], "bookkeeping": [], "accounts payable": [], "accounts receivable": [],
    "general ledger": [], "reconciliation": ["reconciliations"], "financial analysis": [],
    "financial reporting": ["financial statements"], "budgeting": ["budget management"],
    "forecasting": [], "auditing": ["internal audit"], "taxation": ["tax preparation"],
    "gaap": [], "quickbooks": [], "payroll": [], "invoicing": [], "risk management": [],
    "compliance": [], "underwriting": [], "loan processing": ["loan origination"],
    "recruitment": ["recruiting"], "talent acquisition": [], "onboarding": [],
    "employee relations": [], "performance management": [], "benefits administration": [],
    "compensation": [], "hris": [], "training and development": [], "workforce planning": [],
    "sales": [], "business development": [], "lead generation": [], "crm": [],
    "negotiation": [], "account management": [], "customer service": [], "market research": [],
    "seo": [], "social media": [], "content creation": [], "public relations": [],
    "branding": [], "advertising": [], "copywriting": [], "email marketing": [],
    "google analytics": [], "project management": [], "supply chain": [], "logistics": [],
    "inventory management": ["inventory control"], "procurement": [], "strategic planning": [],
    "photoshop": [], "illustrator": [], "indesign": [], "figma": [], "ui/ux": ["ux design", "ui design"],
    "adobe creative suite": ["adobe creative cloud"], "video editing": ["premiere pro"],
    "autocad": [], "solidworks": [], "revit": [], "3d modeling": [],
    "quality control": ["quality assurance"], "six sigma": [], "lean manufacturing": [],
    "project scheduling": [], "blueprint reading": ["blueprints"], "cost estimating": [],
    "osha": [], "safety compliance": [], "preventive maintenance": [], "cad": [],
    "aircraft maintenance": [], "flight operations": [], "faa": [], "vehicle diagnostics": [],
    "patient care": [], "cpr": [], "hipaa": [], "electronic medical records": ["emr", "ehr"],
    "medical terminology": [], "nursing": [], "clinical": [],
    "menu planning": ["menu development"], "food safety": ["servsafe"], "haccp": [],
    "catering": [], "food preparation": [], "kitchen management": [],
    "lesson planning": [], "classroom management": [], "curriculum development": [],
    "tutoring": [], "personal training": [], "nutrition": [], "coaching": [],
    "crop management": [], "irrigation": [],
    "leadership": [], "team management": [], "public speaking": [], "problem solving": [],
}


def _compile():
    pats = {}
    for skill, aliases in SKILLS.items():
        alts = sorted({skill, *aliases}, key=len, reverse=True)
        body = "|".join(re.escape(a) for a in alts)
        pats[skill] = re.compile(rf"(?<![a-z0-9+#.]){'(?:' + body + ')'}(?![a-z+#])")
    return pats

_PATTERNS = _compile()


def extract_skills(text: str) -> list:
    if not isinstance(text, str):
        return []
    t = text.lower()
    return sorted(s for s, p in _PATTERNS.items() if p.search(t))


def fit_score(resume_text: str, jd_text: str, pipe, w_cos=0.6, w_skill=0.4) -> dict:
    tfidf = pipe.named_steps["tfidf"]
    v = tfidf.transform([clean_text(resume_text), clean_text(jd_text)])
    cos = float(cosine_similarity(v[0], v[1])[0, 0])

    resume_sk, jd_sk = set(extract_skills(resume_text)), set(extract_skills(jd_text))
    matched, missing = sorted(resume_sk & jd_sk), sorted(jd_sk - resume_sk)
    overlap = len(matched) / len(jd_sk) if jd_sk else None
    fit = w_cos * cos + w_skill * overlap if overlap is not None else cos
    return {"fit": fit, "cosine": cos, "skill_overlap": overlap,
            "matched_skills": matched, "missing_skills": missing,
            "resume_skills": sorted(resume_sk), "jd_skills": sorted(jd_sk),
            "cosine_only": overlap is None}
