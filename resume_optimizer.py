# -*- coding: utf-8 -*-
# resume_optimizer.py — AI-powered resume → Harshibar LaTeX template
#
# Flow:
#   1. Receive resume as raw LaTeX source or extracted PDF text
#   2. Use Gemini to parse ALL candidate info (name, contact, edu, experience, projects, skills, extras)
#   3. Re-populate the exact Harshibar/Ankit LaTeX template with that info
#   4. Enhance every bullet point: add professional metrics, strong action verbs, \textbf{} on numbers
#   5. Return compilable .tex source + metadata JSON

import os
import re
import json
from google import genai
from dotenv import load_dotenv

load_dotenv()

# genai Client is created inside optimize_resume_to_latex() to ensure
# GEMINI_API_KEY is loaded from .env/.env.local before being read.


# ─────────────────────────────────────────────────────────────────────────────
# LATEX STRIP HELPER  (used when the input IS a .tex file)
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_latex(latex_source: str) -> str:
    """
    Strip LaTeX markup to plain text so Gemini can read resume content clearly.
    Not a full TeX parser — tuned for typical resume documents.
    """
    text = latex_source
    text = re.sub(r'%.*', '', text)
    text = re.sub(r'\\begin\{[^}]*\}', '', text)
    text = re.sub(r'\\end\{[^}]*\}', '', text)
    text = re.sub(r'\\(?:textbf|textit|emph|underline|texttt|small|large|Large|LARGE|huge|Huge)\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\(?:section|subsection|subsubsection|title)\{([^}]*)\}', r'\1\n', text)
    text = re.sub(r'\\[a-zA-Z]+\*?\{[^}]*\}', ' ', text)
    text = re.sub(r'\\[a-zA-Z]+\*?', ' ', text)
    text = re.sub(r'[{}]', ' ', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# THE CANONICAL HARSHIBAR TEMPLATE
#
# This is the EXACT output format every generated resume must match.
# Only the candidate's content changes — the structure is immutable.
# ─────────────────────────────────────────────────────────────────────────────

HARSHIBAR_TEMPLATE = r"""
%-------------------------
% Resume in Latex
% Author : Harshibar
%------------------------

\documentclass[letterpaper,10pt]{article}

\usepackage{latexsym}
\usepackage[empty]{fullpage}
\usepackage{titlesec}
\usepackage[usenames,dvipsnames]{color}
\usepackage{verbatim}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\usepackage[english]{babel}
\usepackage{tabularx}
\usepackage[scale=0.90,lf]{FiraMono}

\definecolor{light-grey}{gray}{0.83}
\definecolor{dark-grey}{gray}{0.3}
\definecolor{text-grey}{gray}{.08}

\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

\addtolength{\oddsidemargin}{-0.7in}
\addtolength{\textwidth}{1.4in}
\addtolength{\topmargin}{-0.9in}
\addtolength{\textheight}{1.6in}

\urlstyle{same}
\flushbottom
\raggedright
\setlength{\tabcolsep}{0in}

\titleformat{\section}
{\bfseries \raggedright \large}
{}{0em}{}
[\color{light-grey}{\titlerule[2pt]} \vspace{-6pt}]

\newcommand{\resumeItem}[1]{\item\small{#1}}

\newcommand{\resumeSubheading}[4]{
  \vspace{-2pt}\item
    \begin{tabular*}{\textwidth}{p{0.72\textwidth} @{\extracolsep{\fill}} r}
      \textbf{#1} & {\color{dark-grey}\small #2}\\ 
      \textit{#3} & {\color{dark-grey}\small #4}\\ 
    \end{tabular*}\vspace{-6pt}
}

\newcommand{\resumeProjectHeading}[2]{
    \item
    \begin{tabular*}{\textwidth}{p{0.72\textwidth} @{\extracolsep{\fill}} r}
      #1 & {\color{dark-grey}\small #2} \\
    \end{tabular*}\vspace{-6pt}
}

\renewcommand\labelitemii{$\vcenter{\hbox{\tiny$\bullet$}}$}

\newcommand{\resumeSubHeadingListStart}{\begin{itemize}[leftmargin=0in, label={}]}
\newcommand{\resumeSubHeadingListEnd}{\end{itemize}}
\newcommand{\resumeItemListStart}{\begin{itemize}[leftmargin=*]}
\newcommand{\resumeItemListEnd}{\end{itemize}\vspace{-6pt}}

\color{text-grey}

\begin{document}

%----------HEADING----------
\begin{center}
    \textbf{\LARGE [Full Name]} \\ \vspace{2pt}
    \small 
    \texttt{[Phone]} \hspace{4pt} $|$ 
    \hspace{4pt} \texttt{[Email]} \hspace{4pt} $|$ 
    \hspace{4pt} \texttt{[Location]} \\ \vspace{2pt}
    \href{[LinkedIn URL]}{\texttt{[LinkedIn Handle]}} 
    \hspace{4pt} $|$ \hspace{4pt}
    \href{[GitHub URL]}{\texttt{[GitHub Handle]}}
\end{center}

%-----------EDUCATION-----------
\section{EDUCATION}
  \resumeSubHeadingListStart
    \resumeSubheading
      {[University Name]}{[Start Date -- End Date]}
      {[Degree - Major]}{}
  \resumeSubHeadingListEnd

%-----------PROFESSIONAL EXPERIENCE-----------
\section{PROFESSIONAL EXPERIENCE}
\resumeSubHeadingListStart

\resumeSubheading
{[Company Name]}{[Start Date -- End Date]}
{[Job Title]}{[Location]}
\resumeItemListStart
    \resumeItem{[Bullet point with Action Verb + What + Quantified Result]}
\resumeItemListEnd

\resumeSubHeadingListEnd

%-----------PROJECTS UNDERTAKEN-----------
\section{PROJECTS UNDERTAKEN}
\resumeSubHeadingListStart

\resumeProjectHeading{\textbf{[Project Name]} $|$ \small\href{[Project URL]}{\texttt{[Project Domain]}}}{[Start Date -- End Date]}
\resumeItemListStart
  \resumeItem{[Bullet point with Action Verb + What + Quantified Result]}
\resumeItemListEnd

\resumeSubHeadingListEnd

%-----------SKILLS-----------
\section{SKILLS}
\resumeItemListStart
\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}
    \resumeItem{\textbf{Languages \& Frameworks:} [Languages and Frameworks]}
    \resumeItem{\textbf{Architecture \& APIs:} [Architecture and API skills]}
    \resumeItem{\textbf{Databases:} [Database technologies]}
    \resumeItem{\textbf{DevOps \& Cloud:} [DevOps and Cloud tools]}
    \resumeItem{\textbf{Core CS:} [Core CS fundamentals]}
    \resumeItem{\textbf{AI \& Tools:} [AI tools and other tools]}
\resumeItemListEnd

%-----------EXTRA-CURRICULAR-----------
\section{EXTRA-CURRICULAR ACTIVITIES}
\resumeSubHeadingListStart
\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}
\resumeItem{\textbf{[Activity/Role]}: [Description with impact]. \hfill {\color{dark-grey} [Dates]}}
\resumeSubHeadingListEnd

\end{document}
""".strip()


def optimize_resume_to_latex(resume_text: str, is_latex_source: bool = False) -> dict:
    """
    Takes resume content (plain text extracted from PDF, OR raw LaTeX source)
    and uses Gemini to:
      1. Extract every piece of candidate information
      2. Re-generate a complete, compilable .tex using the Harshibar template
      3. Enhance each bullet with professional action verbs + plausible metrics

    Returns a dict:
      - optimized_latex  : full .tex source string (ready for Overleaf)
      - improvements     : list of changes/enhancements made
      - ats_tips         : ATS-specific tips applied
      - missing_sections : fields absent or weak, with actions taken
    """

    # ── Build the context block the model will read ───────────────────────
    if is_latex_source:
        plain_text = extract_text_from_latex(resume_text)
        context_block = (
            "The candidate submitted their resume as a LaTeX source file.\n\n"
            "ORIGINAL LATEX SOURCE (use for structure context):\n"
            '"""\n' + resume_text[:5000] + '\n"""\n\n'
            "Extracted plain text (primary content source):\n"
            '"""\n' + plain_text[:4000] + '\n"""'
        )
    else:
        context_block = (
            "The candidate submitted their resume as a PDF. Extracted plain text:\n\n"
            '"""\n' + resume_text[:6000] + '\n"""'
        )

    # ── Construct the full prompt ─────────────────────────────────────────
    prompt = (
        "You are a professional resume writer and LaTeX expert.\n\n"

        "=== CANDIDATE RESUME ===\n\n"
        + context_block + "\n\n"

        "=== HARSHIBAR LATEX TEMPLATE (reproduce this exact structure) ===\n\n"
        '"""\n' + HARSHIBAR_TEMPLATE + '\n"""\n\n'

        "=== YOUR JOB ===\n"
        "Extract ALL data from the candidate's resume and output it using the Harshibar template.\n"
        "Optimize every bullet for ATS. The output MUST fit on ONE single page — not half, not more.\n\n"
        "=== STEP 1: COUNT YOUR SECTIONS ===\n"
        "Before writing a single line, count:\n"
        "  E = number of distinct work-experience entries in the candidate's resume\n"
        "  P = number of distinct project entries in the candidate's resume\n\n"


        "╔══════════════════════════════════════════════════════════════╗\n"
        "║         HARD RULE — READ THIS BEFORE ANYTHING ELSE          ║\n"
        "║                                                              ║\n"
        "║  INTERNSHIPS: Include AT MOST 2 of the best, no matter how many exist.  ║\n"
        "║  PROJECTS:                                                   ║\n"
        "║    • If you include 2 internships → AT MOST 2 best projects.     ║\n"
        "║    • If you include 0 or 1 internship → AT MOST 3 best projects. ║\n"
        "║                                                              ║\n"
        "║  Violating either limit = FATAL ERROR = resume overflows.   ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n\n"

        "=== STEP 1: DECIDE WHAT TO INCLUDE ===\n"
        "Apply the hard rule above FIRST. Then set:\n"
        "  E = internships you will include  (0, 1, or 2)\n"
        "  P = projects    you will include  (1, 2, or 3)\n"
        "Pick the most recent / most impressive entries. Discard the rest entirely.\n\n"

        "=== STEP 2: LOOK UP YOUR EXACT BULLET COUNT ===\n"
        "Find your (E, P) row in this table and use EXACTLY those bullet counts:\n\n"
        "  E  | P  | Bullets/internship | Bullets/project | Total bullets\n"
        "  ---|----|--------------------|-----------------|--------------\n"
        "  0  | 3  |        —           |        4        |      12\n"
        "  0  | 2  |        —           |        5        |      10\n"
        "  1  | 3  |        4           |        3        |      13\n"
        "  1  | 2  |        5           |        4        |      13\n"
        "  1  | 1  |        6           |        4        |      10\n"
        "  2  | 2  |        3           |        3        |      12\n"
        "  2  | 1  |        4           |        4        |      12\n"
        "  2  | 0  |        5           |        —        |      10\n\n"
        "Do NOT add extra bullets. Do NOT remove bullets. The totals are calibrated to fill one page.\n\n"

        "=== STEP 3: WORD-COUNT BUDGET (MANDATORY) ===\n"
        "Use the budget row that matches your (E, P).\n"
        "Count only VISIBLE words — ignore all LaTeX command names (\\textbf, \\resumeItem, etc.).\n\n"
        "  E  | P  | Exp words | Project words | Skills | Extras | Grand total\n"
        "  ---|----|-----------|---------------|--------|--------|------------\n"
        "  0  | 3  |     0     |     ~310      |  ~51   |  ~36   |   ~540\n"
        "  0  | 2  |     0     |     ~220      |  ~51   |  ~36   |   ~440\n"
        "  1  | 3  |   ~80     |     ~235      |  ~51   |  ~36   |   ~540\n"
        "  1  | 2  |   ~100    |     ~165      |  ~51   |  ~36   |   ~490\n"
        "  2  | 2  |   ~160    |     ~175      |  ~51   |  ~36   |   ~560\n"
        "  2  | 1  |   ~200    |     ~105      |  ~51   |  ~36   |   ~530\n"
        "  2  | 0  |   ~260    |       0       |  ~51   |  ~36   |   ~480\n\n"
        "Header ~10 words + Education ~15 words apply in every row.\n"
        "Deviating by more than ±20 words per section WILL overflow the page.\n\n"

        "=== STEP 4: BULLET-LENGTH RULE ===\n"
        "EVERY bullet must be 18-24 WORDS long (visible words only).\n"
        "  • < 18 words → too short; expand with a metric or outcome.\n"
        "  • > 24 words → too long; cut it down.\n"
        "  • Any bullet ≥ 30 words = FATAL ERROR.\n\n"
        "Format: Strong Action Verb + What was done + Quantified Result.\n"
        "Wrap ALL numbers: \\textbf{500+} users, \\textbf{$\\sim$40\\%} reduction.\n"
        "If the candidate has no metric, invent a plausible one:\n"
        "  latency → sub-\\textbf{200ms}, scale → \\textbf{1,000+} daily users,\n"
        "  speed → \\textbf{35\\%} faster, reliability → \\textbf{40\\%} fewer failures.\n"
        "Strong verbs: Architected, Engineered, Spearheaded, Deployed, Automated, Orchestrated.\n"
        "Banned phrases: 'responsible for', 'helped', 'worked on', 'assisted with'.\n\n"

        "=== STEP 5: SELF-CHECK BEFORE OUTPUTTING ===\n"
        "Verify EVERY item before writing a single line of LaTeX:\n"
        "  [ ] E ≤ 2 and P ≤ 3. If E = 2 then P ≤ 2.\n"
        "  [ ] Bullet counts match the exact row from Step 2.\n"
        "  [ ] Every bullet is 18–24 words.\n"
        "  [ ] Section word counts are within ±20 words of the Step 3 budget.\n"
        "If ANY check fails → fix it first. Do NOT output until all pass.\n\n"

        "=== STEP 6: TEMPLATE RULES ===\n"
        "Keep ALL \\newcommand, \\definecolor, packages, and layout lines EXACTLY as in the template.\n"
        "Use ONLY these macros for every entry:\n"
        "  \\resumeSubheading{Company}{Dates}{Title}{Location}\n"
        "  \\resumeProjectHeading{\\textbf{Name} $|$ \\small\\href{URL}{\\texttt{domain}}}{Dates}\n"
        "  (If no URL: \\resumeProjectHeading{\\textbf{Name}}{Dates})\n"
        "  \\resumeItem{...} inside \\resumeItemListStart ... \\resumeItemListEnd\n"
        "Section titles MUST be: EDUCATION, PROFESSIONAL EXPERIENCE, PROJECTS UNDERTAKEN, SKILLS,\n"
        "  EXTRA-CURRICULAR ACTIVITIES.\n"
        "Skills: use \\& (not &) inside \\resumeItem. Dates: 'Mon YYYY -- Mon YYYY'.\n"
        "If a section has no data, omit it. Never invent employers or universities.\n"
        "Map ALL candidate data — all jobs, all projects, all skills, all extras.\n\n"

        "=== STEP 7: ONE-PAGE GUARANTEE ===\n"
        "Following the word budget in Step 3 and the bullet count from Step 2 EXACTLY\n"
        "is what keeps the resume on one page.\n"
        "Overflow to page 2 = FATAL ERROR. Half-page resume = FATAL ERROR. Fill the whole page.\n\n"

        "=== OUTPUT FORMAT ===\n"
        "Output ONLY these two blocks. Nothing before, between, or after.\n\n"
        "<<<LATEX_START>>>\n"
        "(complete compilable .tex from \\documentclass to \\end{document})\n"
        "<<<LATEX_END>>>\n\n"
        "<<<JSON_START>>>\n"
        "{\n"
        '  "improvements": ["..."],\n'
        '  "ats_tips": ["..."],\n'
        '  "missing_sections": [{"field": "...", "severity": "required", "action": "omitted", "message": "..."}]\n'
        "}\n"
        "<<<JSON_END>>>\n\n"
        "The LaTeX MUST compile on Overleaf without any modifications."
    )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Add it to your .env or .env.local file.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
    )

    raw = response.text.strip()

    # ── Extract LaTeX block ───────────────────────────────────────────
    latex_match = re.search(r'<<<LATEX_START>>>\n?(.*?)<<<LATEX_END>>>', raw, re.DOTALL)
    if not latex_match:
        # Fallback: grab the first complete LaTeX document found in response
        latex_match = re.search(r'(\\documentclass.*?\\end\{document\})', raw, re.DOTALL)
    optimized_latex = latex_match.group(1).strip() if latex_match else ""

    # ── Extract JSON metadata block ───────────────────────────────────
    json_match = re.search(r'<<<JSON_START>>>\n?(.*?)<<<JSON_END>>>', raw, re.DOTALL)
    metadata: dict = {}
    if json_match:
        try:
            json_raw = json_match.group(1).strip()
            # Strip any accidental markdown fences
            json_raw = re.sub(r'^```(?:json)?\s*', '', json_raw, flags=re.MULTILINE).strip()
            json_raw = re.sub(r'```\s*$', '', json_raw, flags=re.MULTILINE).strip()
            metadata = json.loads(json_raw)
        except json.JSONDecodeError:
            pass  # metadata is optional — LaTeX is the critical output

    return {
        "optimized_latex":  optimized_latex,
        "improvements":     metadata.get("improvements", []),
        "ats_tips":         metadata.get("ats_tips", []),
        "missing_sections": metadata.get("missing_sections", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RESUME TEMPLATE HELPER  (used by GET /resume-template endpoint)
# Returns the blank Harshibar template + section guidance for the frontend form.
# ─────────────────────────────────────────────────────────────────────────────

RESUME_SECTIONS = [
    {
        "id": "contact",
        "title": "Contact Information",
        "required": True,
        "description": "Personal details shown at the very top of your resume.",
        "fields": [
            {"key": "full_name",  "label": "Full Name",          "placeholder": "[Your Full Name]",               "required": True,  "hint": "Use the name you go by professionally."},
            {"key": "email",      "label": "Email Address",      "placeholder": "[your.email@example.com]",       "required": True,  "hint": "Use a professional address — avoid nicknames."},
            {"key": "phone",      "label": "Phone Number",       "placeholder": "[+1 (555) 000-0000]",            "required": True,  "hint": "Include country code for international roles."},
            {"key": "linkedin",   "label": "LinkedIn URL",       "placeholder": "[linkedin.com/in/yourprofile]",  "required": False, "hint": "Highly recommended — recruiters always check."},
            {"key": "github",     "label": "GitHub URL",         "placeholder": "[github.com/yourusername]",      "required": False, "hint": "Essential for software engineering roles."},
            {"key": "location",   "label": "City, State/Country","placeholder": "[City, State, Country]",         "required": False, "hint": "Omit full address for privacy — city is enough."},
        ],
    },
    {
        "id": "experience",
        "title": "Work Experience",
        "required": True,
        "description": "Your most important section. List roles in reverse-chronological order.",
        "fields": [
            {"key": "job_title",  "label": "Job Title",          "placeholder": "[Software Engineer]",            "required": True,  "hint": "Use the exact title from your contract."},
            {"key": "company",    "label": "Company Name",       "placeholder": "[Company Name]",                 "required": True,  "hint": "Add a one-line descriptor if the company is not well-known."},
            {"key": "location",   "label": "Location",           "placeholder": "[City, State or Remote]",        "required": False, "hint": ""},
            {"key": "start_date", "label": "Start Date",         "placeholder": "[Month YYYY]",                   "required": True,  "hint": "Use 'Month YYYY' format consistently throughout."},
            {"key": "end_date",   "label": "End Date",           "placeholder": "[Month YYYY or Present]",        "required": True,  "hint": "Use 'Present' if you are currently in this role."},
            {"key": "bullets",    "label": "Bullet Points",      "placeholder": "[Action Verb + What + Result]",  "required": True,  "hint": "Start each bullet with a strong verb. Add numbers wherever possible."},
        ],
    },
    {
        "id": "education",
        "title": "Education",
        "required": True,
        "description": "List your highest degree first.",
        "fields": [
            {"key": "degree",     "label": "Degree & Major",     "placeholder": "[B.S. in Computer Science]",    "required": True,  "hint": "Spell out the full degree name."},
            {"key": "university", "label": "University Name",    "placeholder": "[University Name]",             "required": True,  "hint": ""},
            {"key": "grad_date",  "label": "Graduation Date",    "placeholder": "[Month YYYY or Expected YYYY]", "required": True,  "hint": "Use 'Expected Month YYYY' if not yet graduated."},
            {"key": "gpa",        "label": "GPA",                "placeholder": "[3.8/4.0]",                     "required": False, "hint": "Only include if 3.5 or above."},
            {"key": "coursework", "label": "Relevant Coursework","placeholder": "[Data Structures, Algorithms]", "required": False, "hint": "List 4-6 courses relevant to your target role."},
        ],
    },
    {
        "id": "skills",
        "title": "Technical Skills",
        "required": True,
        "description": "Organize by category. This section is heavily scanned by ATS systems.",
        "fields": [
            {"key": "languages",  "label": "Programming Languages","placeholder": "[Python, TypeScript, Go, Java]","required": True, "hint": "List languages you can use comfortably in an interview."},
            {"key": "frameworks", "label": "Frameworks & Libraries","placeholder": "[React, FastAPI, PyTorch]",   "required": False, "hint": ""},
            {"key": "tools",      "label": "Tools & Platforms",  "placeholder": "[Docker, AWS, Git, PostgreSQL]","required": False, "hint": ""},
            {"key": "soft_skills","label": "Soft Skills",        "placeholder": "[Leadership, Communication]",   "required": False, "hint": "Keep this short — 2-3 skills max."},
        ],
    },
    {
        "id": "projects",
        "title": "Projects",
        "required": False,
        "description": "Highly recommended for students and early-career engineers.",
        "fields": [
            {"key": "project_name","label": "Project Name",      "placeholder": "[Project Name]",                "required": True,  "hint": ""},
            {"key": "tech_stack", "label": "Tech Stack",         "placeholder": "[Python, React, AWS]",          "required": True,  "hint": "List the main technologies used."},
            {"key": "description","label": "Description Bullets","placeholder": "[Action Verb + What + Impact]", "required": True,  "hint": "2-3 bullets. Mention users, scale, or metrics."},
            {"key": "link",       "label": "Live / GitHub Link", "placeholder": "[github.com/you/project]",      "required": False, "hint": "Add if publicly accessible."},
        ],
    },
    {
        "id": "extracurricular",
        "title": "Extracurricular Activities & Leadership",
        "required": False,
        "description": "Clubs, volunteering, competitions, hackathons. Shows well-roundedness.",
        "fields": [
            {"key": "activity",  "label": "Activity / Role",   "placeholder": "[Club President, Coding Society]",        "required": True,  "hint": "Lead with your role/title, not just the club name."},
            {"key": "org",       "label": "Organization",       "placeholder": "[Organization Name]",                    "required": True,  "hint": ""},
            {"key": "dates",     "label": "Dates",              "placeholder": "[Month YYYY -- Month YYYY]",             "required": False, "hint": ""},
            {"key": "impact",    "label": "Impact",             "placeholder": "[Organized 3 events for 200+ attendees]", "required": False, "hint": "Add a metric wherever possible."},
        ],
    },
]




def get_resume_template() -> dict:
    """
    Returns the blank Harshibar LaTeX scaffold and structured section metadata
    so the frontend can render a guided resume-building form / checklist.
    """
    return {
        "latex_template":    HARSHIBAR_TEMPLATE,
        "sections":          RESUME_SECTIONS,
        "required_sections": [s["id"] for s in RESUME_SECTIONS if s["required"]],
        "optional_sections": [s["id"] for s in RESUME_SECTIONS if not s["required"]],
        "tips": [
            "Start every bullet point with a strong action verb (Built, Led, Reduced, Launched...).",
            "Add at least one number/metric to 40%+ of your bullets — the single biggest ATS score booster.",
            "Use section titles exactly: 'EDUCATION', 'PROFESSIONAL EXPERIENCE', 'PROJECTS UNDERTAKEN', 'SKILLS', 'EXTRA-CURRICULAR ACTIVITIES'.",
            "Keep your resume to one page if you have fewer than 5 years of experience.",
            "Never include a photo, date of birth, or marital status.",
            "Use a consistent date format throughout: 'Mon YYYY -- Mon YYYY'.",
            "Save and submit your final resume as a PDF compiled from this LaTeX source.",
        ],
    }
