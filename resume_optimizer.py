# resume_optimizer.py — AI-powered resume optimization → LaTeX output

import os
import re
import json
from google import genai
from dotenv import load_dotenv

load_dotenv()

# genai Client is created inside optimize_resume_to_latex() to ensure
# GEMINI_API_KEY is loaded from .env/.env.local before being read.

# ─────────────────────────────────────────────
# LATEX EXTRACTION HELPER
# ─────────────────────────────────────────────

def extract_text_from_latex(latex_source: str) -> str:
    """
    Strip LaTeX commands from source to get plain text for analysis.
    Not a full TeX parser — good enough for resume content extraction.
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


# ─────────────────────────────────────────────
# MAIN OPTIMIZER
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# HARSHIBAR LATEX TEMPLATE (target output format)
# ─────────────────────────────────────────────

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
    Takes resume content (plain text from PDF, or raw LaTeX source) and uses
    Gemini to extract all relevant information, then populates the Harshibar
    LaTeX template with optimized, ATS-friendly content.

    Returns a dict with:
      - optimized_latex  : the full .tex source string
      - improvements     : list of changes made
      - ats_tips         : list of ATS-specific tips applied
      - missing_sections : list of missing/weak fields detected, with action taken
    """

    if is_latex_source:
        plain_text = extract_text_from_latex(resume_text)
        context_block = (
            "The candidate submitted their resume as a LaTeX source file.\n\n"
            "ORIGINAL LATEX SOURCE (for structure reference):\n"
            '"""\n' + resume_text[:4000] + '\n"""\n\n'
            "Extracted plain text (for content extraction):\n"
            '"""\n' + plain_text[:4000] + '\n"""'
        )
    else:
        context_block = (
            "The candidate submitted their resume as a PDF. Extracted plain text:\n\n"
            '"""\n' + resume_text[:5000] + '\n"""'
        )

    prompt = (
        "You are an elite resume writer and LaTeX expert.\n\n"
        "YOUR JOB:\n"
        "1. Extract ALL relevant information from the candidate's resume below.\n"
        "2. Populate the provided Harshibar LaTeX template with that information.\n"
        "3. Optimize every bullet point for ATS and impact.\n"
        "4. Return ONLY the final compilable LaTeX — nothing else.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "CANDIDATE'S RESUME (extract info from this)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        + context_block + "\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "TARGET LATEX TEMPLATE (use this exact structure)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        '"""\n' + HARSHIBAR_TEMPLATE + '\n"""\n\n'
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "OPTIMIZATION RULES\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "1. USE THE HARSHIBAR TEMPLATE EXACTLY — keep all \\newcommand definitions, "
        "\\definecolor, packages, and macros (\\resumeItem, \\resumeSubheading, "
        "\\resumeProjectHeading, \\resumeSubHeadingListStart, etc.) identical.\n"
        "2. EXTRACT all personal info (name, phone, email, location, LinkedIn, GitHub), "
        "education, work experience, projects, skills, extracurriculars from the candidate's resume.\n"
        "3. MAP each piece into the correct section of the Harshibar template.\n"
        "4. BULLET POINT OPTIMIZATION — this is critical:\n"
        "   - Every bullet MUST follow: 'Action Verb + What + Quantified Metric'.\n"
        "   - If a bullet is MISSING numbers (no user count, no latency, no percentage, "
        "no throughput, no time saved), ADD plausible professional metrics. Examples:\n"
        "     • Add user counts: '500+ users', '2,000+ daily active users'\n"
        "     • Add latency: 'sub-200ms response time', 'reduced latency by ~30%'\n"
        "     • Add throughput: '1,000+ API requests/day', '800+ transactions/month'\n"
        "     • Add time savings: 'saving ~15 engineering hours/week'\n"
        "     • Add percentages: 'improved conversion by ~25%', 'reduced errors by ~40%'\n"
        "   - Use \\textbf{} for all metrics/numbers inside bullet points.\n"
        "   - Remove weak phrases: 'responsible for', 'assisted with', 'helped', 'tasked with'.\n"
        "   - Use strong action verbs: Architected, Engineered, Spearheaded, Launched, "
        "Optimized, Streamlined, Integrated, Orchestrated, Deployed, Automated.\n"
        "5. If a section from the candidate's resume doesn't exist in the template, "
        "adapt it into the closest matching section.\n"
        "6. If the candidate is missing a section that exists in the template, OMIT it "
        "entirely — do NOT add placeholder sections.\n"
        "7. Keep the resume to ONE page — be concise but impactful.\n"
        "8. Skills should be categorized exactly as in the template: "
        "Languages & Frameworks, Architecture & APIs, Databases, DevOps & Cloud, Core CS, AI & Tools. "
        "Omit a category if the candidate has nothing for it. Add categories if needed.\n"
        "9. Dates format: 'Mon YYYY -- Mon YYYY' (e.g., 'Jul 2025 -- Sep 2025').\n"
        "10. For projects with live links, use the \\resumeProjectHeading with the URL. "
        "For projects without links, omit the $|$ link part.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "RESPONSE FORMAT\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Output ONLY these two blocks, nothing else:\n\n"
        "<<<LATEX_START>>>\n"
        "(the full, raw, compilable .tex file — from \\documentclass to \\end{document})\n"
        "<<<LATEX_END>>>\n\n"
        "<<<JSON_START>>>\n"
        "{\n"
        '  "improvements": ["change 1", "change 2"],\n'
        '  "ats_tips": ["tip 1", "tip 2"],\n'
        '  "missing_sections": [\n'
        '    {"field": "name", "severity": "required|recommended", "action": "added_placeholder|added_section|flagged", "message": "friendly instruction"}\n'
        "  ]\n"
        "}\n"
        "<<<JSON_END>>>\n\n"
        "No text before, between, or after the two blocks. "
        "The LaTeX MUST be complete and compilable on Overleaf with no modifications."
    )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Add it to your .env.local file.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )

    raw = response.text.strip()

    # ── Extract LaTeX block ───────────────────────────────────────────
    latex_match = re.search(r'<<<LATEX_START>>>\n?(.*?)<<<LATEX_END>>>', raw, re.DOTALL)
    if not latex_match:
        # Fallback: grab anything that looks like a LaTeX document
        latex_match = re.search(r'(\\documentclass.*?\\end\{document\})', raw, re.DOTALL)
    optimized_latex = latex_match.group(1).strip() if latex_match else ""

    # ── Extract JSON metadata block ───────────────────────────────────
    json_match = re.search(r'<<<JSON_START>>>\n?(.*?)<<<JSON_END>>>', raw, re.DOTALL)
    metadata: dict = {}
    if json_match:
        try:
            json_raw = json_match.group(1).strip()
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


# ─────────────────────────────────────────────
# RESUME TEMPLATE GENERATOR
# ─────────────────────────────────────────────

# Section metadata returned alongside the LaTeX template so the
# client can render a guided form / checklist.
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
        "id": "certifications",
        "title": "Certifications",
        "required": False,
        "description": "Add relevant industry certifications (AWS, Google Cloud, PMP, CPA, etc.).",
        "fields": [
            {"key": "cert_name",  "label": "Certification Name","placeholder": "[AWS Certified Solutions Architect]","required": True, "hint": ""},
            {"key": "issuer",     "label": "Issuing Organization","placeholder": "[Amazon Web Services]",        "required": True,  "hint": ""},
            {"key": "cert_date",  "label": "Date",               "placeholder": "[Month YYYY]",                  "required": False, "hint": ""},
        ],
    },
    {
        "id": "extracurricular",
        "title": "Extracurricular Activities & Leadership",
        "required": False,
        "description": "Clubs, volunteering, competitions, societies. Shows well-roundedness.",
        "fields": [
            {"key": "activity",   "label": "Activity / Role",   "placeholder": "[Club President, Coding Society]","required": True, "hint": "Lead with your role/title, not just the club name."},
            {"key": "org",        "label": "Organization",       "placeholder": "[Organization Name]",            "required": True,  "hint": ""},
            {"key": "dates",      "label": "Dates",              "placeholder": "[Month YYYY -- Month YYYY]",     "required": False, "hint": ""},
            {"key": "impact",     "label": "Impact",             "placeholder": "[Organized 3 events for 200+ attendees]","required": False, "hint": "Add a metric wherever possible."},
        ],
    },
    {
        "id": "achievements",
        "title": "Achievements & Awards",
        "required": False,
        "description": "Hackathon wins, academic honors, scholarships, competitions.",
        "fields": [
            {"key": "achievement","label": "Achievement",        "placeholder": "[Award Name, Organization, Year]","required": True, "hint": "'1st place out of 200 teams' is stronger than just 'Winner'."},
        ],
    },
]

LATEX_TEMPLATE = r"""\documentclass[letterpaper,11pt]{article}

% ── Packages ──────────────────────────────────────────────────────
\usepackage[top=0.6in, bottom=0.6in, left=0.75in, right=0.75in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{xcolor}
\usepackage{parskip}

\hypersetup{colorlinks=true, urlcolor=blue, linkcolor=black}

\titleformat{\section}{\large\bfseries}{}{0em}{}[\titlerule]
\titlespacing{\section}{0pt}{8pt}{4pt}
\setlist[itemize]{leftmargin=1.5em, itemsep=1pt, parsep=0pt, topsep=2pt}
\pagestyle{empty}

% ══════════════════════════════════════════════════════════════════
\begin{document}

% ── CONTACT INFORMATION (REQUIRED) ────────────────────────────────
\begin{center}
  {\LARGE\bfseries [Your Full Name]} \\[4pt]
  \href{mailto:[your.email@example.com]}{[your.email@example.com]}
  \quad|\quad [+1 (555) 000-0000]
  \quad|\quad [City, State] \\[2pt]
  \href{https://[linkedin.com/in/yourprofile]}{LinkedIn}
  \quad|\quad
  \href{https://[github.com/yourusername]}{GitHub}
\end{center}

% ── WORK EXPERIENCE (REQUIRED) ────────────────────────────────────
\section{Experience}

\textbf{[Job Title]} \hfill [Month YYYY] -- [Month YYYY or Present] \\
\textit{[Company Name]} \hfill [City, State or Remote]
\begin{itemize}
  \item [Action verb + what you did + measurable result.]
  \item [Action verb + what you did + measurable result.]
\end{itemize}

% ── EDUCATION (REQUIRED) ──────────────────────────────────────────
\section{Education}

\textbf{[University Name]} \hfill [Month YYYY] \\
\textit{[B.S. / M.S. in Your Major]} \hfill [City, State]
\begin{itemize}
  \item GPA: [3.X/4.0]
  \item Relevant Coursework: [Course 1, Course 2, Course 3]
\end{itemize}

% ── TECHNICAL SKILLS (REQUIRED) ───────────────────────────────────
\section{Technical Skills}

\begin{itemize}[leftmargin=0pt, label={}]
  \item \textbf{Languages:}   [Python, TypeScript, Go, Java, C++]
  \item \textbf{Frameworks:}  [React, FastAPI, Node.js, PyTorch]
  \item \textbf{Tools:}       [Docker, Kubernetes, AWS, Git, PostgreSQL]
\end{itemize}

% ── PROJECTS (RECOMMENDED) ────────────────────────────────────────
\section{Projects}

\textbf{[Project Name]} \hfill \href{https://[github.com/you/project]}{GitHub}
\begin{itemize}
  \item Built with: [Python, React, AWS, PostgreSQL]
  \item [Describe impact / scale.]
\end{itemize}

% ── EXTRACURRICULAR ACTIVITIES (RECOMMENDED) ──────────────────────
\section{Extracurricular Activities}

\textbf{[Role / Position]} \hfill [Month YYYY] -- [Month YYYY] \\
\textit{[Organization Name]}
\begin{itemize}
  \item [Describe what you did and its impact.]
\end{itemize}

% ── CERTIFICATIONS (OPTIONAL) ─────────────────────────────────────
\section{Certifications}

\begin{itemize}
  \item \textbf{[Certification Name]} -- [Issuing Organization], [Month YYYY]
\end{itemize}

% ── ACHIEVEMENTS (OPTIONAL) ───────────────────────────────────────
\section{Achievements}

\begin{itemize}
  \item [Award name, Organization, Year.]
\end{itemize}

\end{document}
"""


def get_resume_template() -> dict:
    """
    Returns a LaTeX resume scaffold with placeholder text and
    a structured list of sections with field-level guidance.
    """
    return {
        "latex_template":    LATEX_TEMPLATE,
        "sections":          RESUME_SECTIONS,
        "required_sections": [s["id"] for s in RESUME_SECTIONS if s["required"]],
        "optional_sections": [s["id"] for s in RESUME_SECTIONS if not s["required"]],
        "tips": [
            "Start every bullet point with a strong action verb (Built, Led, Reduced, Launched...).",
            "Add at least one number/metric to 40%+ of your bullets — the single biggest ATS score booster.",
            "Use exact section titles: 'Experience', 'Education', 'Skills', 'Projects' — ATS parsers look for these.",
            "Keep your resume to one page if you have fewer than 5 years of experience.",
            "Never include a photo, date of birth, or marital status.",
            "Use a consistent date format throughout: 'Month YYYY -- Month YYYY'.",
            "Save and submit your final resume as a PDF compiled from this LaTeX source.",
        ],
    }
