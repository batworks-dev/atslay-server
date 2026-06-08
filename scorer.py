# scorer.py — ATSlay scoring logic extracted from notebook

import re
import nltk
import textstat
import pdfplumber
import docx
from pathlib import Path
from collections import Counter
from nltk.corpus import stopwords

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
        'python','java','javascript','typescript','c++','c#','ruby','go','rust',
        'swift','kotlin','php','r','scala','matlab','bash','perl','html','css',
        'react','angular','vue','node','django','flask','spring','fastapi',
        'tensorflow','pytorch','keras','pandas','numpy','scikit-learn','opencv',
        'aws','azure','gcp','docker','kubernetes','terraform','ansible','jenkins',
        'ci/cd','github actions','linux','nginx','redis','rabbitmq',
        'machine learning','deep learning','nlp','computer vision','data science',
        'data analysis','sql','nosql','mongodb','postgresql','mysql','spark',
        'hadoop','airflow','tableau','power bi','looker','dbt',
        'git','jira','confluence','figma','excel','rest api','graphql',
        'microservices','agile','scrum','devops','blockchain','salesforce',
        'project management','stakeholder management','cross-functional',
        'budget management','strategic planning','market research',
        'financial analysis','risk management','supply chain','crm',
        'digital marketing','seo','content strategy','ux','ui',
        'product management','data-driven','kpi','roi','p&l',
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
        'award','recognition','promoted','top performer','exceeded','outperformed',
        'record','milestone','breakthrough','patented','published','speaker',
        'president','vice president','director','founder','co-founder','head of',
        'lead','principal','senior','captain','chairman','secretary general',
        'finalist','winner','champion','hackathon','selected','scholarship',
        'national','international','ranked','top','honor','distinction','merit',
        'participants','members','volunteers','clients','users','customers',
        'deployed','live','production','launched','shipped',
        'vision','roadmap','strategy','innovative','transformation','disruption',
        'scalable','enterprise','end-to-end','full-stack','full stack',
        'revenue','profit','savings','cost reduction','efficiency','productivity',
        'growth','retention','acquisition','conversion','engagement',
        'certified','accredited','licensed','awarded','invited',
        'mentored','coached','presented','authored','contributed',
    ]

    FILLER_WORDS = [
        'responsible for','duties included','worked on','helped with',
        'assisted in','involved in','participated in','tasked with',
        'was in charge of','handled','did','made','used','utilized',
        'leverage','leveraged','passionate','dynamic','synergy','proactive',
        'go-getter','results-oriented','team player','hardworking',
        'detail-oriented','self-starter','thought leader','guru','ninja',
        'rock star','wizard','evangelist','visionary',
    ]

    UNNECESSARY_SECTIONS = [
        'objective','references','references available','hobbies','interests',
        'personal interests','personal profile','date of birth','dob',
        'nationality','marital status','gender','religion','passport',
        'photograph','photo','age','height','weight','blood group',
        'father name','mother name','permanent address','declaration',
        'i hereby declare','i declare that',
    ]

    GOOD_SECTIONS = [
        'experience','work experience','employment history','professional experience',
        'education','academic background','skills','technical skills',
        'projects','certifications','summary','professional summary',
        'achievements','awards','publications','languages',
        'volunteer','leadership','extracurricular',
    ]

    DATE_PATTERNS = {
        'good': [
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s,]+\d{4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)[\s,]+\d{4}\b',
            r'\b\d{4}\s*[-–]\s*(?:\d{4}|Present|Current|Now)\b',
            r'\b(?:0?[1-9]|1[0-2])[/\-]\d{4}\b',
        ],
        'bad': [
            r'\b(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/\d{2,4}\b',
            r'\b(?:0?[1-9]|[12]\d|3[01])\.(?:0?[1-9]|1[0-2])\.\d{2,4}\b',
            r"\b'\d{2}\b",
            r'\b\d{2}/\d{2}\b',
        ]
    }

    ACTION_VERBS = [
        'led','managed','directed','supervised','oversaw','spearheaded','championed',
        'orchestrated','headed','coordinated','delegated','mentored','coached',
        'achieved','delivered','exceeded','surpassed','accomplished','attained',
        'generated','produced','created','built','launched','shipped','deployed',
        'improved','optimized','streamlined','automated','accelerated','enhanced',
        'upgraded','restructured','revamped','transformed','modernized','redesigned',
        'grew','increased','boosted','expanded','scaled','doubled','tripled',
        'reduced','cut','eliminated','saved','lowered','decreased',
        'collaborated','partnered','liaised','negotiated','facilitated','aligned',
        'analyzed','evaluated','assessed','identified','researched','investigated',
        'diagnosed','audited','reviewed','monitored','tracked','measured',
        'presented','authored','published','documented','reported','communicated',
        'trained','educated','facilitated','conducted','organized','planned',
        'developed','engineered','designed','implemented','integrated','configured',
        'tested','debugged','deployed','migrated','architected','programmed',
        'established','founded','pioneered','initiated','introduced','executed',
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

    STRONG_QUANT_PATTERNS = [
        r'\d+\s*%',
        r'\$[\d,]+[kKmMbB]?\b',
        r'\d+[kKmMbB]\+?\s*(?:users|requests|api|calls|records|events|participants)',
        r'\d{1,3},\d{3}',
        r'(?:doubled|tripled|halved|10x|5x|3x|2x)',
        r'99\.9%|100%',
        r'(?:cut|reduced|decreased|lowered|eliminated|shrunk).{0,40}\d+',
        r'(?:increased|grew|boosted|improved|raised|scaled).{0,40}\d+',
        r'\d+\+?\s*(?:team members|engineers|developers|employees|people)',
        r'\d+\s*(?:seconds?|minutes?|hours?|days?)\s*(?:faster|reduction|saved|less)',
        r'(?:slashing|cutting|reducing).{0,30}(?:from|by).{0,30}\d+',
        r'(?:top|ranked|#)\s*\d+\b',
    ]

    def extract_all(self, text: str) -> dict:
        tl = text.lower()
        lines = text.split('\n')
        words = re.findall(r'\b[a-z][a-z0-9\+\#\.]*\b', tl)
        word_freq = Counter(w for w in words if w not in STOP_WORDS and len(w) > 2)

        return {
            'tech_keywords_found':   self._find_keywords(tl),
            'domain_hits':           self._find_domain_keywords(tl),
            'total_keyword_count':   len(self._find_keywords(tl)),
            'buzzwords_found':       self._find_buzzwords(tl),
            'filler_words_found':    self._find_fillers(tl),
            'repetitive_words':      self._find_repetitive_words(word_freq),
            'good_sections_found':   self._find_good_sections(tl),
            'unnecessary_sections':  self._find_unnecessary_sections(tl),
            'sections_missing':      self._find_missing_sections(tl),
            'flesch_score':          self._flesch_score(text),
            'avg_sentence_len':      self._avg_sentence_len(text),
            'bullet_ratio':          self._bullet_ratio(lines),
            'long_bullets':          self._find_long_bullets(lines),
            'word_count':            len(text.split()),
            'good_date_formats':     self._find_dates(text, 'good'),
            'bad_date_formats':      self._find_dates(text, 'bad'),
            'has_present_marker':    bool(re.search(r'\b(?:present|current|now|ongoing|continuing)\b', tl) or re.search(r'202[5-9]|203\d', text)),
            'action_verbs_found':    self._find_action_verbs(tl),
            'soft_skills_found':     self._find_soft_skills(tl),
            'has_quantification':    self._has_quantification(text),
            '_quant_result':         self._count_quantified(text),
            'quantified_count':      self._count_quantified(text)[0],
            'total_bullets':         self._count_quantified(text)[1],
            'unquantified_bullets':  self._count_quantified(text)[3][:8],
            'email_found':           bool(re.search(r'[\w.+-]+@[\w-]+\.[a-z]{2,}', text)),
            'phone_found':           bool(re.search(r'[\+\d][\d\s\-\(\)]{8,}', text)),
            'linkedin_found':        'linkedin' in tl,
            'github_found':          'github' in tl,
        }

    def _find_keywords(self, t):
        return list(set(k for k in self.TECH_KEYWORDS if re.search(r'\b' + re.escape(k) + r'\b', t)))

    def _find_domain_keywords(self, t):
        return {d: [k for k in kws if k in t] for d, kws in self.DOMAIN_KEYWORDS.items() if any(k in t for k in kws)}

    def _find_buzzwords(self, t):
        return [b for b in self.BUZZWORDS if re.search(r'\b' + re.escape(b) + r'\b', t)]

    def _find_fillers(self, t):
        return [f for f in self.FILLER_WORDS if f in t]

    def _find_repetitive_words(self, freq):
        WHITELIST = {
            'python','javascript','typescript','react','node','flask','django','docker',
            'kubernetes','postgresql','mongodb','mysql','redis','linux','github',
            'java','aws','azure','gcp','git','sql','html','css','api',
            'integrated','implemented','developed','deployed','automated','engineered',
            'designed','architected','optimized','streamlined','launched','delivered',
            'built','created','managed','led','configured','tested','analyzed',
            'automation','validation','pipeline','workflow','system','platform',
            'service','endpoint','application','database','framework','backend',
            'frontend','interface','module','feature','function','component',
            'using','based','driven','oriented','focused','related','specific',
            'team','work','project','company','role','position','year','experience',
            'data','entry','parking','payment','feature','request','build','test',
            'error','code','stack','layer','level','type','time','load','page',
        }
        return {w: c for w, c in freq.items() if c >= 5 and w not in WHITELIST and len(w) > 5}

    def _find_good_sections(self, t):
        return [s for s in self.GOOD_SECTIONS if s in t]

    def _find_unnecessary_sections(self, t):
        return [s for s in self.UNNECESSARY_SECTIONS if s in t]

    def _find_missing_sections(self, t):
        found = self._find_good_sections(t)
        return [s for s in ['experience', 'education', 'skills'] if not any(s in f for f in found)]

    def _flesch_score(self, text):
        try:
            return textstat.flesch_reading_ease(text)
        except:
            return 50.0

    def _avg_sentence_len(self, text):
        try:
            sentences = nltk.sent_tokenize(text)
            return sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        except:
            return 20

    def _bullet_ratio(self, lines):
        bullet_lines = [l for l in lines if re.match(r'^\s*[•\-\*▪►●◦▸]', l.strip())]
        content_lines = [l for l in lines if len(l.strip()) > 20]
        return len(bullet_lines) / max(1, len(content_lines))

    def _find_long_bullets(self, lines):
        return [l.strip() for l in lines if re.match(r'^\s*[•\-\*▪►●◦▸]', l.strip()) and len(l.split()) > 30]

    def _find_dates(self, text, kind):
        results = []
        for pattern in self.DATE_PATTERNS[kind]:
            for m in re.findall(pattern, text, re.IGNORECASE):
                results.append(m if isinstance(m, str) else m[0])
        return results

    def _find_action_verbs(self, t):
        return [v for v in self.ACTION_VERBS if re.search(r'\b' + v + r'\b', t)]

    def _find_soft_skills(self, t):
        return [s for s in self.SOFT_SKILLS if s in t]

    def _has_quantification(self, text):
        return any(re.search(p, text, re.IGNORECASE) for p in self.STRONG_QUANT_PATTERNS)

    def _extract_bullets(self, text):
        bullets = []
        for line in text.split('\n'):
            s = line.strip()
            if re.match(r'^[-*]', s) and len(s) > 15:
                bullets.append(s.lstrip('-* ').strip())
            elif len(s) > 30 and re.match(r'^[A-Z][a-z]', s) and s.endswith('.') and len(s.split()) > 6:
                bullets.append(s)
        return bullets

    def _count_quantified(self, text):
        bullets = self._extract_bullets(text) or [l.strip() for l in text.split('\n') if len(l.strip()) > 25]
        quantified, unquantified = [], []
        for b in bullets:
            if any(re.search(p, b, re.IGNORECASE) for p in self.STRONG_QUANT_PATTERNS):
                quantified.append(b)
            else:
                unquantified.append(b)
        return len(quantified), len(bullets), quantified, unquantified


# ─────────────────────────────────────────────
# ATS SCORER
# ─────────────────────────────────────────────

class ATSScorer:
    WEIGHTS = {
        'keywords': 0.15, 'buzzwords': 0.08, 'repetition': 0.07,
        'sections': 0.15, 'readability': 0.10, 'dates': 0.05, 'communication': 0.40,
    }

    def score(self, f: dict) -> dict:
        critical_sections = {'experience', 'education', 'skills'}
        missing_critical = [s for s in f['sections_missing'] if s.lower() in critical_sections]

        if missing_critical:
            return {
                'final_score': 25,
                'dimension_scores': {'keywords': 10, 'buzzwords': 15, 'repetition': 40,
                                     'sections': 0, 'readability': 50, 'dates': 40, 'communication': 10},
                'gate_triggered': f'Missing critical section(s): {missing_critical}',
            }

        if f.get('flesch_score', 50) < 25:
            return {
                'final_score': 35,
                'dimension_scores': {'keywords': 30, 'buzzwords': 25, 'repetition': 30,
                                     'sections': 60, 'readability': 15, 'dates': 50, 'communication': 25},
                'gate_triggered': f"Readability failure: Flesch={f['flesch_score']} (unreadable)",
            }

        scores = {
            'keywords':      self._score_keywords(f),
            'buzzwords':     self._score_buzzwords(f),
            'repetition':    self._score_repetition(f),
            'sections':      self._score_sections(f),
            'readability':   self._score_readability(f),
            'dates':         self._score_dates(f),
            'communication': self._score_communication(f),
        }
        final = round(min(100, max(0, sum(scores[d] * self.WEIGHTS[d] for d in scores))))
        return {'final_score': final, 'dimension_scores': {k: round(v) for k, v in scores.items()}, 'gate_triggered': None}

    def _score_keywords(self, f):
        c = f['total_keyword_count']
        if c == 0:    return 10
        elif c <= 3:  return 25 + c * 8
        elif c <= 7:  return 50 + c * 5
        elif c <= 12: return 85 + c * 2
        else:         return min(100, 95 + c // 5)

    def _score_buzzwords(self, f):
        c = len(f['buzzwords_found'])
        if c == 0:   return 20
        elif c <= 1: return 40
        elif c <= 3: return 55 + c * 5
        elif c <= 5: return 70 + c * 4
        else:        return min(100, 85 + c)

    def _score_repetition(self, f):
        return max(10, 100 - len(f['filler_words_found']) * 12 - len(f['repetitive_words']) * 10)

    def _score_sections(self, f):
        return max(0, 100 - len(f['unnecessary_sections']) * 15 - len(f['sections_missing']) * 25)

    def _score_readability(self, f):
        flesch = f['flesch_score']
        fs = 100 if flesch >= 70 else 90 if flesch >= 60 else 75 if flesch >= 50 else 60 if flesch >= 40 else 40 if flesch >= 30 else 20
        avg = f['avg_sentence_len']
        ss = 100 if 12 <= avg <= 22 else 80 if 10 <= avg <= 28 else 50
        lbp = max(0, 100 - len(f['long_bullets']) * 15)
        return fs * 0.70 + ss * 0.15 + lbp * 0.15

    def _score_dates(self, f):
        good, bad = len(f['good_date_formats']), len(f['bad_date_formats'])
        if good == 0 and bad == 0: return 60
        return max(10, min(100, 70 + min(15, good * 5) - bad * 20 + (10 if f['has_present_marker'] else 0)))

    def _score_communication(self, f):
        verbs, quant = len(f['action_verbs_found']), f['quantified_count']
        total_b = max(1, f.get('total_bullets', 1))
        ratio = quant / total_b
        qs = 100 if ratio >= 0.8 else 85 if ratio >= 0.6 else 65 if ratio >= 0.4 else 40 if ratio >= 0.2 else 20 if ratio >= 0.1 else 0
        if quant < 2: qs = min(qs, 35)
        vs = 95 if verbs >= 15 else 85 if verbs >= 12 else 75 if verbs >= 10 else 60 if verbs >= 8 else 40 if verbs >= 5 else 20 if verbs >= 2 else 5
        vs = max(5, vs - len(f['filler_words_found']) * 15)
        soft = len(f['soft_skills_found'])
        ss = 100 if soft >= 8 else 90 if soft >= 6 else 75 if soft >= 4 else 55 if soft >= 2 else 35 if soft >= 1 else 10
        return qs * 0.60 + vs * 0.30 + ss * 0.10


# ─────────────────────────────────────────────
# SUGGESTIONS ENGINE
# ─────────────────────────────────────────────

class SuggestionsEngine:
    def generate(self, features: dict, scores: dict) -> list:
        suggestions = []
        dims = scores['dimension_scores']
        f = features

        if dims['keywords'] < 75:
            suggestions.append({'priority': 'HIGH', 'category': 'Data & Keywords',
                'issue': f"Only {f['total_keyword_count']} technical/domain keywords detected.",
                'fix': 'Add specific tools, technologies, and domain terms. Mirror exact terminology from job postings.',
                'impact': f"+{min(20, (75 - dims['keywords']) // 2)} pts potential"})

        if dims['buzzwords'] < 60:
            suggestions.append({'priority': 'MEDIUM', 'category': 'Buzzwords & Power Words',
                'issue': f"Only {len(f['buzzwords_found'])} industry buzzwords found.",
                'fix': 'Add impact words: "exceeded targets", "revenue growth", "award-winning", "promoted", "certified".',
                'impact': '+10-15 pts potential'})

        if dims['repetition'] < 75:
            if f['filler_words_found']:
                suggestions.append({'priority': 'HIGH', 'category': 'Filler / Weak Phrases',
                    'issue': f"Weak phrases found: {', '.join(f['filler_words_found'][:5])}",
                    'fix': 'Replace passive fillers with strong action verbs. Use "Managed" instead of "responsible for managing".',
                    'impact': f"-{len(f['filler_words_found']) * 8} pts currently"})
            if f['repetitive_words']:
                suggestions.append({'priority': 'MEDIUM', 'category': 'Repetitive Words',
                    'issue': f"Overused words: {', '.join(list(f['repetitive_words'].keys())[:5])}",
                    'fix': 'Vary your vocabulary — use synonyms and different action verbs.',
                    'impact': f"-{len(f['repetitive_words']) * 5} pts currently"})

        if f['unnecessary_sections']:
            suggestions.append({'priority': 'HIGH', 'category': 'Unnecessary Sections',
                'issue': f"ATS-harmful sections detected: {', '.join(f['unnecessary_sections'])}",
                'fix': 'Remove: Objective, References, DOB, Hobbies, Marital Status, Declaration, Photograph.',
                'impact': f"-{len(f['unnecessary_sections']) * 15} pts currently"})

        if f['sections_missing']:
            suggestions.append({'priority': 'HIGH', 'category': 'Missing Critical Sections',
                'issue': f"Required sections not found: {', '.join(f['sections_missing'])}",
                'fix': f"Add clear section headers: {', '.join(s.upper() for s in f['sections_missing'])}.",
                'impact': f"-{len(f['sections_missing']) * 20} pts currently"})

        if dims['readability'] < 70:
            issues = []
            if f['avg_sentence_len'] > 25: issues.append(f"sentences average {f['avg_sentence_len']:.0f} words (aim 15-20)")
            if len(f['long_bullets']) > 0: issues.append(f"{len(f['long_bullets'])} bullet(s) exceed 30 words")
            if f['word_count'] < 300:      issues.append(f"too short ({f['word_count']} words, aim 400-800)")
            if issues:
                suggestions.append({'priority': 'MEDIUM', 'category': 'Readability',
                    'issue': '; '.join(issues),
                    'fix': 'Keep bullets to 1-2 lines. Split long sentences. Aim for 450-800 words total.',
                    'impact': '+10-15 pts potential'})

        if dims['dates'] < 70:
            suggestions.append({'priority': 'MEDIUM', 'category': 'Date Formatting',
                'issue': f"Inconsistent date formats: {f['bad_date_formats'][:3]}",
                'fix': 'Use: "Jan 2022 – Present" or "2020 – 2023". Avoid DD/MM/YY or ambiguous formats.',
                'impact': '+8-12 pts potential'})

        if dims['communication'] < 70:
            if len(f['action_verbs_found']) < 8:
                suggestions.append({'priority': 'HIGH', 'category': 'Action Verbs',
                    'issue': f"Only {len(f['action_verbs_found'])} strong action verbs (target: 10+).",
                    'fix': 'Start every bullet with: Led, Built, Grew, Reduced, Launched, Delivered, Optimized.',
                    'impact': '+15-20 pts potential'})
            total_b = f.get('total_bullets', 0)
            quant = f['quantified_count']
            ratio_pct = round((quant / total_b * 100) if total_b else 0)
            if ratio_pct < 50 or quant < 3:
                suggestions.append({'priority': 'HIGH' if ratio_pct < 30 else 'MEDIUM',
                    'category': 'Quantified Achievements',
                    'issue': f"Only {quant} of {total_b} bullets quantified ({ratio_pct}%). Target: 50%+.",
                    'fix': 'Add metrics: "Reduced load time by 40%", "Processed 10K daily requests", "Served 5,000+ users".',
                    'impact': f"+{min(25, (50 - ratio_pct) // 3)} pts potential"})

        if not f['email_found']:
            suggestions.append({'priority': 'HIGH', 'category': 'Contact Info', 'issue': 'No email detected.', 'fix': 'Add email at the top.', 'impact': 'Critical'})
        if not f['phone_found']:
            suggestions.append({'priority': 'HIGH', 'category': 'Contact Info', 'issue': 'No phone number detected.', 'fix': 'Add phone number.', 'impact': 'Critical'})
        if not f['linkedin_found']:
            suggestions.append({'priority': 'LOW', 'category': 'Online Presence', 'issue': 'No LinkedIn URL found.', 'fix': 'Add your LinkedIn profile URL.', 'impact': '+5 pts'})

        order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        suggestions.sort(key=lambda x: order.get(x['priority'], 3))
        return suggestions


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

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

    if len(clean) < 50:
        return {'error': 'Could not extract text. Is the PDF image-based or scanned?'}

    features    = extractor.extract_all(clean)
    scores      = scorer.score(features)
    suggestions = suggester.generate(features, scores)

    # Make features JSON-serialisable (remove internal tuple)
    features.pop('_quant_result', None)
    features['repetitive_words'] = dict(features.get('repetitive_words', {}))

    return {
        'ats_score':        scores['final_score'],
        'grade':            _grade(scores['final_score']),
        'gate_triggered':   scores.get('gate_triggered'),
        'dimension_scores': scores['dimension_scores'],
        'features':         features,
        'suggestions':      suggestions,
    }

def _grade(score):
    if score >= 90: return 'SLAYING'
    if score >= 80: return 'STRONG'
    if score >= 70: return 'GOOD'
    if score >= 55: return 'AVERAGE'
    if score >= 40: return 'NEEDS WORK'
    return 'CRITICAL'