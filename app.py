# app.py — Flask HTTP server for ATSlay

import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from scorer import analyse_resume
from jd_scorer import score_resume_against_jd
from scorer import ResumeParser
from dotenv import load_dotenv
load_dotenv()  

app = Flask(__name__)

# Where uploaded files are temporarily saved
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB max upload

os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # create folder if it doesn't exist


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/score-resume', methods=['POST'])
def score_resume():
    # ── 1. Check a file was sent ──────────────────────────────────
    if 'resume' not in request.files:
        return jsonify({'error': 'No file found. Send file with key "resume"'}), 400

    file = request.files['resume']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use PDF or DOCX'}), 400

    # ── 2. Save file temporarily ──────────────────────────────────
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # ── 3. Run the ATSlay scoring ─────────────────────────────────
    try:
        result = analyse_resume(save_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Always delete the uploaded file after processing
        if os.path.exists(save_path):
            os.remove(save_path)

    # ── 4. Return JSON response ───────────────────────────────────
    return jsonify(result), 200


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ATSlay API is running'}), 200


@app.route('/score-resume-jd', methods=['POST'])
def score_resume_jd():

    # ── 1. Validate resume file ───────────────────────────────────
    if 'resume' not in request.files:
        return jsonify({'error': 'No resume file. Send file with key "resume"'}), 400

    resume_file = request.files['resume']
    if resume_file.filename == '':
        return jsonify({'error': 'No resume file selected'}), 400
    if not allowed_file(resume_file.filename):
        return jsonify({'error': 'Invalid resume type. Use PDF or DOCX'}), 400

    # ── 2. Validate JD file ───────────────────────────────────────
    if 'jd' not in request.files:
        return jsonify({'error': 'No JD file. Send file with key "jd"'}), 400

    jd_file = request.files['jd']
    if jd_file.filename == '':
        return jsonify({'error': 'No JD file selected'}), 400
    if not allowed_file(jd_file.filename):
        return jsonify({'error': 'Invalid JD type. Use PDF, DOCX, or TXT'}), 400

    # ── 3. Save both files temporarily ───────────────────────────
    resume_filename = secure_filename(resume_file.filename)
    jd_filename     = secure_filename(jd_file.filename)

    resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_filename)
    jd_path     = os.path.join(app.config['UPLOAD_FOLDER'], jd_filename)

    resume_file.save(resume_path)
    jd_file.save(jd_path)

    try:
        parser = ResumeParser()

        # ── 4. Extract resume text ────────────────────────────────
        resume_text = parser.clean_text(parser.extract_text(resume_path))
        if len(resume_text) < 100:
            return jsonify({'error': 'Could not extract text from resume.'}), 400

        # ── 5. Extract JD text ────────────────────────────────────
        jd_ext = jd_path.rsplit('.', 1)[1].lower()

        if jd_ext == 'txt':
            with open(jd_path, 'r', encoding='utf-8') as f:
                jd_text = f.read().strip()
        else:
            # PDF or DOCX — reuse the same parser
            jd_text = parser.clean_text(parser.extract_text(jd_path))

        if len(jd_text) < 50:
            return jsonify({'error': 'Could not extract text from JD file.'}), 400

        # ── 6. Run Gemini JD matching ─────────────────────────────
        jd_result = score_resume_against_jd(resume_text, jd_text)

        # ── 7. Run standard ATS score alongside ──────────────────
        standard_result = analyse_resume(resume_path)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(resume_path):
            os.remove(resume_path)
        if os.path.exists(jd_path):
            os.remove(jd_path)

    # ── 8. Return merged result ───────────────────────────────────
    return jsonify({
        'jd_analysis': jd_result,
        'general_ats': {
            'ats_score':        standard_result.get('ats_score'),
            'dimension_scores': standard_result.get('dimension_scores'),
            'suggestions':      standard_result.get('suggestions'),
        }
    }), 200




if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)