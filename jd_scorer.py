# jd_scorer.py — JD-based ATS scoring using Google Gemini

import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURE GEMINI
# Set GEMINI_API_KEY in your environment or .env file
# Get free key at: https://aistudio.google.com/app/apikey
# ─────────────────────────────────────────────

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


def score_resume_against_jd(resume_text: str, jd_text: str) -> dict:
    """
    Uses Gemini to compare a resume against a job description.
    Returns a structured JSON with match score, gaps, and suggestions.
    """

    prompt = f"""
You are an expert ATS (Applicant Tracking System) and career coach.

Analyze the resume below against the job description and return a JSON object only — no extra text, no markdown, no code fences.

RESUME:
\"\"\"
{resume_text[:4000]}
\"\"\"

JOB DESCRIPTION:
\"\"\"
{jd_text[:2000]}
\"\"\"

Return ONLY this JSON structure (fill in all fields accurately):

{{
  "jd_match_score": <integer 0–100, overall match percentage>,
  "dimension_scores": {{
    "skills_match": <0–100, how many required skills are present>,
    "experience_match": <0–100, years/level alignment>,
    "keyword_overlap": <0–100, JD keywords found in resume>,
    "role_alignment": <0–100, how well the role matches past titles/work>
  }},
  "matched_keywords": [<list of JD keywords found in resume>],
  "missing_keywords": [<list of important JD keywords NOT in resume>],
  "matched_skills": [<list of required skills the candidate has>],
  "missing_skills": [<list of required skills the candidate lacks>],
  "experience_gap": "<string: describe any experience gap or 'No gap detected'>",
  "strengths": [<3–5 bullet strings of what makes the candidate a strong fit>],
  "improvements": [
    {{
      "priority": "<HIGH | MEDIUM | LOW>",
      "category": "<Skills | Keywords | Experience | Formatting>",
      "issue": "<what is missing or weak>",
      "fix": "<specific actionable fix to improve JD match>"
    }}
  ],
  "hiring_recommendation": "<STRONG FIT | GOOD FIT | PARTIAL FIT | NOT A FIT>",
  "summary": "<2–3 sentence plain-English summary of the match>"
}}
"""

    model = genai.GenerativeModel("gemini-3.5-flash")  # free-tier model
    response = model.generate_content(prompt)

    raw = response.text.strip()

    # Strip markdown fences if Gemini adds them anyway
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    result = json.loads(raw)
    return result