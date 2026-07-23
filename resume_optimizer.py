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
\raggedbottom
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
        "You are a world-class resume writer and LaTeX expert specializing in ATS optimization.\n\n"

        "════════════════════════════════════════\n"
        "YOUR TASK\n"
        "════════════════════════════════════════\n\n"
        "1. Carefully read the candidate's resume below.\n"
        "2. Extract EVERY piece of information: name, phone, email, location, LinkedIn, GitHub,\n"
        "   education (university, degree, dates), ALL work experiences, ALL projects,\n"
        "   ALL skills, and ALL extracurricular/hackathon entries.\n"
        "3. Re-generate the resume using the EXACT Harshibar LaTeX template structure provided.\n"
        "4. Enhance bullet points to be professional and ATS-optimized.\n"
        "5. Return ONLY the LaTeX output block + metadata JSON block — nothing else.\n\n"

        "════════════════════════════════════════\n"
        "CANDIDATE'S RESUME\n"
        "════════════════════════════════════════\n\n"
        + context_block + "\n\n"

        "════════════════════════════════════════\n"
        "HARSHIBAR LATEX TEMPLATE  <- USE THIS EXACT STRUCTURE\n"
        "════════════════════════════════════════\n\n"
        '"""\n' + HARSHIBAR_TEMPLATE + '\n"""\n\n'

        "════════════════════════════════════════\n"
        "STRICT RULES — READ CAREFULLY\n"
        "════════════════════════════════════════\n\n"

        "TEMPLATE FIDELITY (non-negotiable):\n"
        "  * Keep ALL \\newcommand definitions, \\definecolor, package imports, and layout commands\n"
        "    EXACTLY as they appear in the Harshibar template. Do not alter them.\n"
        "  * Use \\resumeSubheading, \\resumeProjectHeading, \\resumeSubHeadingListStart,\n"
        "    \\resumeSubHeadingListEnd, \\resumeItemListStart, \\resumeItemListEnd, \\resumeItem\n"
        "    for EVERY entry — exactly as shown in the template.\n"
        "  * Section titles must match exactly: EDUCATION, PROFESSIONAL EXPERIENCE,\n"
        "    PROJECTS UNDERTAKEN, SKILLS, EXTRA-CURRICULAR ACTIVITIES.\n\n"

        "CONTENT MAPPING:\n"
        "  * Map ALL candidate information into the correct template sections.\n"
        "  * If the candidate has multiple jobs, repeat \\resumeSubheading blocks inside\n"
        "    \\resumeSubHeadingListStart / \\resumeSubHeadingListEnd.\n"
        "  * If the candidate has multiple projects, repeat \\resumeProjectHeading blocks.\n"
        "  * If a section is completely absent from the candidate's resume, OMIT that section entirely.\n"
        "  * Do NOT invent companies, universities, or roles not present in the original resume.\n\n"

        "BULLET POINT ENHANCEMENT (critical for ATS):\n"
        "  * Every bullet MUST follow: Action Verb + What was done + Quantified Result.\n"
        "  * Wrap ALL metrics/numbers in \\textbf{}: e.g., \\textbf{99.9\\% uptime},\n"
        "    \\textbf{1,000+ daily API requests}, \\textbf{$\\sim$30\\%}.\n"
        "  * If a bullet is missing numbers, ADD plausible professional metrics. Examples:\n"
        "      - User counts:  '\\textbf{500+} registered users', '\\textbf{2,000+} daily active users'\n"
        "      - Latency:      'sub-\\textbf{200ms} response time', 'reducing latency by \\textbf{$\\sim$30\\%}'\n"
        "      - Throughput:   '\\textbf{1,000+} API requests/day', '\\textbf{800+} transactions/month'\n"
        "      - Time savings: 'saving \\textbf{$\\sim$15} engineering hours/week'\n"
        "      - Error rates:  'reducing errors by \\textbf{$\\sim$40\\%}'\n"
        "      - DB perf:      'cutting query execution time by \\textbf{70\\%}'\n"
        "      - Test coverage:'achieving \\textbf{95\\%} automated test coverage'\n"
        "  * Remove weak phrases: 'responsible for', 'assisted with', 'helped', 'worked on', 'tasked with'.\n"
        "  * Use strong action verbs: Architected, Engineered, Spearheaded, Launched, Optimized,\n"
        "    Streamlined, Integrated, Orchestrated, Deployed, Automated, Reduced, Accelerated.\n"
        "  * Provide 3 bullet points per experience/project (never fewer than 2).\n\n"

        "PROJECT LINK FORMATTING:\n"
        "  * If a project has a live URL use:\n"
        "    \\resumeProjectHeading{\\textbf{Name} $|$ \\small\\href{URL}{\\texttt{domain}}}{dates}\n"
        "  * If no URL, use:\n"
        "    \\resumeProjectHeading{\\textbf{Name}}{dates}\n\n"

        "SKILLS SECTION:\n"
        "  * Keep the six categories exactly as in the template:\n"
        "    Languages & Frameworks, Architecture & APIs, Databases, DevOps & Cloud,\n"
        "    Core CS, AI & Tools.\n"
        "  * Omit a category only if the candidate has zero skills for it.\n"
        "  * Use \\& (not bare &) inside \\resumeItem text.\n\n"

        "EXTRA-CURRICULAR / HACKATHONS:\n"
        "  * Use the \\resumeItem{\\textbf{...}: description. \\hfill {\\color{dark-grey} dates}}\n"
        "    pattern shown in the template.\n"
        "  * Include ALL clubs, hackathons, competitions, and leadership roles found.\n\n"

        "DATES: Format 'Mon YYYY -- Mon YYYY' (e.g., 'Jul 2025 -- Sep 2025'). Use 'Present' for ongoing.\n\n"

        "PAGE LENGTH: Target ONE page. Be concise but impactful.\n\n"

        "════════════════════════════════════════\n"
        "OUTPUT FORMAT — EXACT STRUCTURE REQUIRED\n"
        "════════════════════════════════════════\n\n"
        "Output ONLY the two blocks below. No extra text before, between, or after.\n\n"
        "<<<LATEX_START>>>\n"
        "(complete, raw, compilable .tex -- from \\documentclass to \\end{document})\n"
        "<<<LATEX_END>>>\n\n"
        "<<<JSON_START>>>\n"
        "{\n"
        '  "improvements": ["enhancement 1", "enhancement 2", "..."],\n'
        '  "ats_tips": ["tip 1", "tip 2", "..."],\n'
        '  "missing_sections": [\n'
        '    {"field": "field_name", "severity": "required|recommended",\n'
        '     "action": "omitted|flagged", "message": "friendly note for user"}\n'
        "  ]\n"
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
