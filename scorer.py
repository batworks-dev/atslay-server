# scorer.py — ATSlay scoring logic extracted from notebook

import re
import nltk
import textstat
import pdfplumber
import docx
from pathlib import Path
from collections import Counter
from nltk.corpus import stopwords

# Download NLTK data at startup (needed on Render — build and runtime are separate environments)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

STOP_WORDS = set(stopwords.words('english'))


# ─────────────────────────────────────────────
# RESUME PARSER
# ─────────────────────────────────────────────

class ResumeParser:
    def extract_text(self, file_path: str) -> str:
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext == '.pdf':
            return self._from_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            return self._from_docx(file_path)
        else:
            raise ValueError(f"Unsupported format: {ext}. Use PDF or DOCX.")

    def _from_pdf(self, path):
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text.strip()

    def _from_docx(self, path):
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()


# ─────────────────────────────────────────────
# FEATURE EXTRACTOR
# ─────────────────────────────────────────────

class FeatureExtractor:
    TECH_KEYWORDS = [
        'python', 'java', 'javascript', 'typescript', 'c++', 'go', 'rust',
        'react', 'next.js', 'node.js', 'express', 'flask', 'django',
        'mongodb', 'postgresql', 'mysql', 'firebase', 'redis',
        'aws', 'ec2', 's3', 'cloudfront', 'docker', 'kubernetes', 'vercel',
        'git', 'github', 'gitlab', 'ci/cd', 'jenkins',
        'langgraph', 'langchain', 'claude', 'anthropic', 'openai',
        'machine learning', 'ai', 'llm', 'prompt engineering',
        'tailwind css', 'shadcn', 'html5', 'css3', 'rest api', 'graphql',
        'cashfree', 'payment integration', 'qr code', 'attendance system'
    ]

    DOMAIN_KEYWORDS = {
        'tech':    ['algorithm','debugging','deployment','version control','testing','api'],
        'finance': ['financial modeling','valuation','portfolio','derivatives','equity','audit'],
        'marketing': ['campaign','brand','conversion','engagement','funnel','analytics'],
        'hr':      ['recruitment','onboarding','talent','performance review','compliance','payroll'],
        'design':  ['wireframe','prototype','user research','usability','adobe','sketch'],
        'ops':     ['logistics','procurement','vendor','process improvement','lean','six sigma'],
    }

    BUZZWORDS = [
        'president', 'finalist', 'hackathon', 'led', 'launched', 'shipped',
        'deployed', 'built', 'engineered', 'orchestrated', 'automated',
        'optimized', 'reduced', 'improved', 'increased', 'accelerated',
        'real-time', 'end-to-end', 'full-stack', 'autonomous', 'scalable',
        'award', 'winner', 'top', 'achieved', 'delivered'
    ]

    FILLER_WORDS = [
        'responsible for', 'duties included', 'assisted in', 'helped with',
        'participated in', 'tasked with', 'was in charge of', 'handled'
    ]

    UNNECESSARY_SECTIONS = [
        'objective', 'references', 'hobbies', 'date of birth', 'dob',
        'nationality', 'marital status', 'religion', 'photograph'
    ]

    GOOD_SECTIONS = [
        'professional experience', 'work experience', 'experience',
        'education', 'skills', 'technical skills', 'projects',
        'certifications', 'achievements', 'publications',
        'extracurricular activities', 'leadership'
    ]
    
    ACTION_VERBS = [
        'led', 'managed', 'developed', 'built', 'created', 'designed',
        'implemented', 'deployed', 'launched', 'shipped', 'engineered',
        'orchestrated', 'automated', 'optimized', 'streamlined', 'reduced',
        'improved', 'increased', 'accelerated', 'integrated', 'configured',
        'tested', 'debugged', 'architected', 'delivered', 'achieved',
        'spearheaded', 'championed', 'coordinated', 'mentored'
    ]

    SOFT_SKILLS = [
        'communication','leadership','teamwork','collaboration','problem-solving',
        'problem solving','analytical','creative','detail-oriented','adaptable',
        'organised','organized','time management','critical thinking','interpersonal',
        'initiative','innovative','motivated','flexible','strategic','empathetic',
        'persuasive','negotiation','conflict resolution','decision-making',
        'multitasking','prioritization','stakeholder','cross-functional',
        'client-facing','presentation skills','written communication',
        'verbal communication','active listening',
        'code review','code reviews','clean code','version control',
        'best practices','agile methodology','scrum methodology',
        'continuous improvement','ownership','accountability',
        'attention to detail','fast learner','quick learner','self-motivated',
        'independent','proactive','driven','dedicated','passionate',
        'collaborative','result-oriented','results-driven',
    ]

    def extract_all(self, text: str) -> dict:
        tl = text.lower()
        lines = text.split('\n')

        # Extract words for frequency analysis
        words = re.findall(r'\b[a-z][a-z0-9\+#\.]*\b', tl)
        word_freq = Counter(w for w in words if w not in STOP_WORDS and len(w) > 2)

        # Check for sections (case insensitive)
        has_experience = any(s in tl for s in ['professional experience', 'work experience', 'experience'])
        has_education = 'education' in tl
        has_skills = 'skills' in tl

        # IMPORTANT FIX: Check if sections exist properly
        sections_missing = []
        if not has_experience:
            sections_missing.append('experience')
        if not has_education:
            sections_missing.append('education')
        if not has_skills:
            sections_missing.append('skills')

        # Count true bullet points (not headers or contact info)
        true_bullets = self._extract_true_bullets(lines)
        quantified_count, unquantified = self._count_quantified_bullets(true_bullets)

        # Readability with adjusted calculation
        flesch = self._get_flesch_score(text)
        avg_sentence_len = self._get_avg_sentence_len(text)

        return {
            # Keywords
            'tech_keywords_found': self._find_keywords(tl),
            'total_keyword_count': len(self._find_keywords(tl)),

            # Buzzwords
            'buzzwords_found': self._find_buzzwords(tl),

            # Repetition
            'filler_words_found': self._find_fillers(tl),
            'repetitive_words': self._find_repetitive_words(word_freq),

            # Sections (FIXED)
            'good_sections_found': self._find_good_sections(tl),
            'unnecessary_sections': self._find_unnecessary_sections(tl),
            'sections_missing': sections_missing,
            'has_experience': has_experience,
            'has_education': has_education,
            'has_skills': has_skills,

            # Readability
            'flesch_score': flesch,
            'avg_sentence_len': avg_sentence_len,
            'word_count': len(text.split()),
            'long_sentences': self._count_long_sentences(text),

            # Dates
            'good_date_formats': self._find_good_dates(text),
            'has_present_marker': bool(re.search(r'\b(present|current|now|ongoing)\b', tl)),

            # Communication (FIXED - accurate bullet counting)
            'action_verbs_found': self._find_action_verbs(tl),
            'soft_skills_found': self._find_soft_skills(tl),
            'quantified_count': quantified_count,
            'total_bullets': len(true_bullets),
            'unquantified_bullets': unquantified[:5],
            'bullet_ratio': quantified_count / max(1, len(true_bullets)),

            # Contact
            'email_found': bool(re.search(r'[\w.+-]+@[\w-]+\.[a-z]{2,}', text)),
            'phone_found': bool(re.search(r'[\+\d][\d\s\-\(\)]{8,}', text)),
            'linkedin_found': 'linkedin' in tl,
            'github_found': 'github' in tl,
        }
        
    def _find_keywords(self, t):
        found = []
        for kw in self.TECH_KEYWORDS:
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, t):
                found.append(kw)
        return list(set(found))

    def _find_buzzwords(self, t):
        found = []
        for bw in self.BUZZWORDS:
            if re.search(r'\b' + re.escape(bw) + r'\b', t):
                found.append(bw)
        return found

    def _find_fillers(self, t):
        return [f for f in self.FILLER_WORDS if f in t]

    def _find_repetitive_words(self, freq):
        # Only flag truly excessive repetition (10+ times)
        return {w: c for w, c in freq.items() if c >= 10 and len(w) > 4}

    def _find_good_sections(self, t):
        return [s for s in self.GOOD_SECTIONS if s in t]

    def _find_unnecessary_sections(self, t):
        return [s for s in self.UNNECESSARY_SECTIONS if s in t]

    def _get_flesch_score(self, text):
        try:
            score = textstat.flesch_reading_ease(text)
            # Cap at reasonable range for technical resumes
            return max(20, min(80, score))
        except:
            return 50

    def _get_avg_sentence_len(self, text):
        try:
            sentences = nltk.sent_tokenize(text)
            if not sentences:
                return 20
            # Ignore very short sentences (like headers)
            valid_sentences = [s for s in sentences if len(s.split()) > 5]
            if not valid_sentences:
                return 20
            return sum(len(s.split()) for s in valid_sentences) / len(valid_sentences)
        except:
            return 20

    def _count_long_sentences(self, text):
        try:
            sentences = nltk.sent_tokenize(text)
            return sum(1 for s in sentences if len(s.split()) > 35)
        except:
            return 0

    def _extract_true_bullets(self, lines):
        """Extract only real bullet points, ignoring headers and contact info"""
        bullets = []
        for line in lines:
            stripped = line.strip()
            # Skip empty or very short lines
            if len(stripped) < 15:
                continue
            # Skip lines that look like headers (all caps or ends with colon)
            if stripped.isupper() or stripped.endswith(':'):
                continue
            # Skip contact info lines
            if '@' in stripped or 'linkedin' in stripped.lower() or 'github' in stripped.lower():
                continue
            if re.match(r'^[\+\d]', stripped):  # Phone numbers
                continue
            # Check if it's a bullet point (starts with bullet symbol or dash)
            if re.match(r'^[\s]*[•\-*▪►●◦▸]', stripped):
                bullets.append(re.sub(r'^[\s]*[•\-*▪►●◦▸]\s*', '', stripped))
            # Also include lines that start with action verbs (likely resume content)
            elif re.match(r'^(Led|Developed|Built|Engineered|Implemented|Deployed|Launched|Orchestrated|Automated|Optimized|Reduced|Improved|Integrated|Created|Designed|Architected|Delivered|Managed)', stripped, re.I):
                bullets.append(stripped)
        return bullets

    def _count_quantified_bullets(self, bullets):
        """Count bullets that have quantifiable metrics"""
        quant_patterns = [
            r'\d+\s*%',
            r'\$[\d,]+[kKmMbB]?',
            r'\d+[kKmMbB]\s*(?:users|requests|api|calls|records|events)',
            r'\d{1,3},\d{3}',
            r'(?:reduced|cut|decreased|lowered|eliminated).{0,30}\d+',
            r'(?:increased|grew|boosted|improved|scaled).{0,30}\d+',
            r'\d+\+?\s*(?:team|member|engineer|developer)s?',
            r'\d+\s*(?:seconds?|minutes?|hours?|days?)\s*(?:faster|reduction)',
            r'99\.9%|100%',
            r'(?:10x|5x|3x|2x|doubled|tripled)'
        ]

        quantified = []
        unquantified = []

        for bullet in bullets:
            is_quantified = any(re.search(p, bullet, re.IGNORECASE) for p in quant_patterns)
            if is_quantified:
                quantified.append(bullet)
            else:
                unquantified.append(bullet)

        return len(quantified), unquantified

    def _find_good_dates(self, text):
        patterns = [
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s,]+(?:20|19)\d{2}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)[\s,]+(?:20|19)\d{2}\b',
            r'\b(?:20|19)\d{2}\s*[–-]\s*(?:(?:20|19)\d{2}|Present|Current)\b'
        ]
        dates = []
        for pattern in patterns:
            dates.extend(re.findall(pattern, text, re.IGNORECASE))
        return list(set(dates))

    def _find_action_verbs(self, t):
        return [v for v in self.ACTION_VERBS if re.search(r'\b' + v + r'\b', t)]

    def _find_soft_skills(self, t):
        soft = ['communication', 'leadership', 'teamwork', 'problem-solving', 'analytical']
        return [s for s in soft if s in t]


# ─────────────────────────────────────────────
# ATS SCORER
# ─────────────────────────────────────────────

class ATSScorer:
    """
    FIXED: Realistic ATS scoring that accurately evaluates strong resumes
    """

    def score(self, f: dict) -> dict:
        # Check for critical issues (only truly missing sections)
        missing_critical = f.get('sections_missing', [])

        # If resume actually has experience but we missed it, override
        if f.get('has_experience', False) and 'experience' in missing_critical:
            missing_critical.remove('experience')
        if f.get('has_education', False) and 'education' in missing_critical:
            missing_critical.remove('education')
        if f.get('has_skills', False) and 'skills' in missing_critical:
            missing_critical.remove('skills')

        # Only penalize if truly missing critical sections
        if missing_critical and len(missing_critical) >= 2:
            return {
                'final_score': 40,
                'dimension_scores': self._get_low_scores(),
                'gate_triggered': f'Missing: {missing_critical}'
            }

        # Normal scoring
        scores = {
            'keywords': self._score_keywords(f),
            'buzzwords': self._score_buzzwords(f),
            'repetition': self._score_repetition(f),
            'sections': self._score_sections(f),
            'readability': self._score_readability(f),
            'dates': self._score_dates(f),
            'communication': self._score_communication(f),
        }

        # Weighted final score (adjusted weights)
        weights = {
            'keywords': 0.20,
            'buzzwords': 0.10,
            'repetition': 0.05,
            'sections': 0.15,
            'readability': 0.10,
            'dates': 0.05,
            'communication': 0.35,
        }

        final = sum(scores[d] * weights[d] for d in scores)
        final = round(min(100, max(0, final)))

        return {
            'final_score': final,
            'dimension_scores': {k: round(v) for k, v in scores.items()},
            'gate_triggered': None,
        }

    def _get_low_scores(self):
        return {
            'keywords': 40, 'buzzwords': 40, 'repetition': 60,
            'sections': 30, 'readability': 50, 'dates': 50, 'communication': 35
        }

    def _score_keywords(self, f):
        count = f['total_keyword_count']
        if count >= 20: return 95
        if count >= 15: return 85
        if count >= 10: return 70
        if count >= 5: return 50
        return 30

    def _score_buzzwords(self, f):
        count = len(f['buzzwords_found'])
        if count >= 10: return 90
        if count >= 7: return 75
        if count >= 4: return 55
        if count >= 2: return 40
        return 25

    def _score_repetition(self, f):
        fillers = len(f['filler_words_found'])
        rep_words = len(f['repetitive_words'])
        score = 85
        score -= fillers * 8
        score -= rep_words * 5
        return max(40, min(100, score))

    def _score_sections(self, f):
        missing = len([m for m in f.get('sections_missing', [])])
        unnecessary = len(f['unnecessary_sections'])

        score = 90
        score -= missing * 20
        score -= unnecessary * 10
        return max(30, min(100, score))

    def _score_readability(self, f):
        flesch = f['flesch_score']
        avg_sent = f['avg_sentence_len']
        long_sent = f.get('long_sentences', 0)

        # Flesch (lower threshold for technical resumes)
        if flesch >= 50: fs = 90
        elif flesch >= 40: fs = 75
        elif flesch >= 30: fs = 60
        elif flesch >= 20: fs = 50
        else: fs = 40

        # Sentence length (technical resumes can have longer sentences)
        if avg_sent <= 25: ss = 90
        elif avg_sent <= 30: ss = 75
        elif avg_sent <= 35: ss = 60
        else: ss = 45

        # Penalize excessive long sentences
        long_penalty = max(0, long_sent * 3)

        return max(40, min(100, fs * 0.6 + ss * 0.4 - long_penalty))

    def _score_dates(self, f):
        good = len(f['good_date_formats'])
        has_present = f['has_present_marker']

        if good >= 3:
            score = 85
        elif good >= 1:
            score = 65
        else:
            score = 40

        if has_present:
            score += 10

        return min(100, score)

    def _score_communication(self, f):
        verbs = len(f['action_verbs_found'])
        quant_ratio = f.get('bullet_ratio', 0)
        total_bullets = f.get('total_bullets', 1)

        # Verb score
        if verbs >= 12: vs = 90
        elif verbs >= 8: vs = 75
        elif verbs >= 5: vs = 55
        elif verbs >= 3: vs = 40
        else: vs = 25

        # Quantification ratio (what matters most)
        if quant_ratio >= 0.5: qs = 100
        elif quant_ratio >= 0.35: qs = 85
        elif quant_ratio >= 0.25: qs = 70
        elif quant_ratio >= 0.15: qs = 50
        else: qs = 30

        # Boost for having at least 2 quantified bullets
        if f.get('quantified_count', 0) >= 3:
            qs = min(100, qs + 10)

        return vs * 0.4 + qs * 0.6

# ─────────────────────────────────────────────
# SUGGESTIONS ENGINE
# ─────────────────────────────────────────────

class SuggestionsEngine:
    
    def generate(self, features: dict, scores: dict) -> list:
        suggestions = []
        dims = scores.get('dimension_scores', {})

        # Keywords
        if dims.get('keywords', 100) < 70:
            suggestions.append({
                'priority': '🟡 MEDIUM',
                'category': 'Keywords',
                'issue': 'Add more technical keywords relevant to your target role.',
                'fix': 'Review job descriptions and include specific tools, frameworks, and technologies.',
                'impact': 'High'
            })

        # Communication/Quantification
        quant_ratio = features.get('bullet_ratio', 0)
        if quant_ratio < 0.3:
            suggestions.append({
                'priority': '🔴 HIGH',
                'category': 'Quantified Achievements',
                'issue': f'Only {features.get("quantified_count", 0)} of {features.get("total_bullets", 1)} bullets have metrics.',
                'fix': 'Add numbers: "Reduced X by Y%", "Processed Z requests", "Managed $N budget"',
                'impact': 'Critical - Add metrics to 40%+ of bullets'
            })

        # Action verbs
        verbs = len(features.get('action_verbs_found', []))
        if verbs < 8:
            suggestions.append({
                'priority': '🔴 HIGH',
                'category': 'Action Verbs',
                'issue': f'Only {verbs} strong action verbs found.',
                'fix': 'Start each bullet with: Led, Built, Launched, Optimized, Reduced, Delivered',
                'impact': 'High'
            })

        # Readability
        if features.get('avg_sentence_len', 0) > 35:
            suggestions.append({
                'priority': '🟡 MEDIUM',
                'category': 'Readability',
                'issue': 'Some sentences are too long.',
                'fix': 'Break long sentences into shorter, punchier statements.',
                'impact': 'Medium'
            })

        # Unnecessary sections
        if features.get('unnecessary_sections'):
            suggestions.append({
                'priority': '🟢 LOW',
                'category': 'Sections',
                'issue': f'Unnecessary sections: {features["unnecessary_sections"]}',
                'fix': 'Remove personal details (age, marital status, religion, photo).',
                'impact': 'Low'
            })

        return suggestions
# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

class ReportGenerator:
    def print_report(self, resume_path, features, scores, suggestions):
        final = scores['final_score']
        dims = scores['dimension_scores']

        # Grade
        if final >= 85: grade, emoji = "EXCELLENT", "🟢"
        elif final >= 70: grade, emoji = "GOOD", "🟢"
        elif final >= 55: grade, emoji = "AVERAGE", "🟡"
        elif final >= 40: grade, emoji = "NEEDS WORK", "🔴"
        else: grade, emoji = "CRITICAL", "🔴"

        print("\n" + "="*70)
        print("  🔥 ATSlay AI — Resume Analysis Report")
        print("="*70)
        print(f"  📄 File     : {Path(resume_path).name}")
        print(f"  📊 Score    : {final}/100  {emoji}  {grade}")
        print("="*70)

        print("\n📊 DIMENSION SCORES")
        print("-"*50)

        dim_names = {
            'keywords': 'Keywords & Tech Stack',
            'buzzwords': 'Achievement Buzzwords',
            'repetition': 'Word Repetition',
            'sections': 'Section Quality',
            'readability': 'Readability',
            'dates': 'Date Formatting',
            'communication': 'Communication & Impact'
        }

        for dim, name in dim_names.items():
            score = dims.get(dim, 0)
            bar = "█" * (score // 5) + "░" * (20 - (score // 5))
            print(f"  {name:<25} [{bar}] {score:3d}%")

        print("\n📈 KEY METRICS")
        print("-"*50)
        print(f"  Technical Keywords : {features.get('total_keyword_count', 0)}")
        print(f"  Action Verbs       : {len(features.get('action_verbs_found', []))}")
        print(f"  Buzzwords          : {len(features.get('buzzwords_found', []))}")
        print(f"  Quantified Bullets : {features.get('quantified_count', 0)}/{features.get('total_bullets', 1)} ({int(features.get('bullet_ratio', 0)*100)}%)")
        print(f"  Word Count         : {features.get('word_count', 0)}")
        print(f"  Avg Sentence Length: {features.get('avg_sentence_len', 0):.1f} words")

        print("\n💡 TOP IMPROVEMENTS")
        print("-"*50)
        if suggestions:
            for i, s in enumerate(suggestions[:5], 1):
                print(f"\n  [{i}] {s['priority']} — {s['category']}")
                print(f"      → {s['issue']}")
                print(f"      ✓ {s['fix']}")
        else:
            print("  🎉 Your resume is in great shape!")

        print("\n" + "="*70)
        print(f"  Final Score: {final}/100  |  Target: 85+")
        print("="*70 + "\n")

print("✅ ReportGenerator ready!")


# ============================================
# CELL 8 - Main Engine
# ============================================
class ATSlayEngine:
    def __init__(self):
        self.parser = ResumeParser()
        self.extractor = FeatureExtractor()
        self.scorer = ATSScorer()
        self.suggester = SuggestionsEngine()
        self.reporter = ReportGenerator()

    def analyse(self, resume_path: str):
        print(f"\n⏳ Analysing: {Path(resume_path).name}...")

        # Extract text
        raw = self.parser.extract_text(resume_path)
        clean = self.parser.clean_text(raw)

        if len(clean) < 100:
            print("❌ Could not extract text properly.")
            return None

        # Extract features
        features = self.extractor.extract_all(clean)

        # Score
        scores = self.scorer.score(features)

        # Generate suggestions
        suggestions = self.suggester.generate(features, scores)

        # Report
        self.reporter.print_report(resume_path, features, scores, suggestions)

        return {
            'ats_score': scores['final_score'],
            'dimensions': scores['dimension_scores'],
            'features': features,
            'suggestions': suggestions
        }

print("✅ ATSlayEngine ready!")


def analyse_resume(file_path: str) -> dict:
    """
    Main function called by Flask.
    Takes a file path, returns a JSON-serialisable dict.
    """
    parser    = ResumeParser()
    extractor = FeatureExtractor()
    scorer    = ATSScorer()
    suggester = SuggestionsEngine()

    raw   = parser.extract_text(file_path)
    clean = parser.clean_text(raw)

    if len(clean) < 100:
        return {'error': 'Could not extract text. Is the PDF image-based or scanned?'}

    features    = extractor.extract_all(clean)
    scores      = scorer.score(features)
    suggestions = suggester.generate(features, scores)

    return {
        'ats_score':        scores['final_score'],
        'dimension_scores': scores['dimension_scores'],
        'features':         features,
        'suggestions':      suggestions,
    }
