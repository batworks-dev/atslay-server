# app.py — Flask HTTP server for ATSlay

import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from scorer import analyse_resume

app = Flask(__name__)

# Where uploaded files are temporarily saved
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB max upload


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


# NEW (works on Render too)
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)