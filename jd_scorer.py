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
    Returns a concise, recruiter-friendly JSON with match score and key insights.
    """

    prompt = f"""
You are an expert ATS (Applicant Tracking System) and career coach.

Analyze the resume below against the job description. Return ONLY a JSON object — no extra text, no markdown, no code fences.

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
  "matching_score": <integer 0–100, overall match percentage>,
  "recommendation": "<STRONG FIT | GOOD FIT | PARTIAL FIT | NOT A FIT>",
  "strengths": ["<top 3–5 concise strengths relevant to this JD>"],
  "gaps": ["<top 3–5 concise skill/experience gaps>"],
  "top_improvements": ["<top 3–5 specific, actionable improvement suggestions>"]
}}

RULES:
- matching_score must be an integer between 0 and 100.
- recommendation must be one of: STRONG FIT, GOOD FIT, PARTIAL FIT, NOT A FIT.
- Each array must have 3 to 5 items maximum.
- Keep each item short (one sentence max).
- Do NOT include dimension breakdowns, keyword lists, or verbose summaries.
"""

    model = genai.GenerativeModel("gemini-2.0-flash")  # free-tier model
    response = model.generate_content(prompt)

    raw = response.text.strip()

    # Strip markdown fences if Gemini adds them anyway
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    result = json.loads(raw)
    return result