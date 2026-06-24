# auth.py — JWT authentication middleware for ATSlay

import os
import re
from functools import wraps
from flask import request, jsonify, g
import jwt
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
# JWT CONFIG
# ─────────────────────────────────────────────

JWT_SECRET = os.environ.get("JWT_PASSWORD")
JWT_ALGORITHM = "HS256"

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$")


def require_auth(f):
    """
    Decorator that verifies the JWT from the Authorization header.
    Extracts and validates the email field from the decoded payload.
    Attaches the full decoded payload to `g.jwt_payload` and email to `g.email`.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        # ── 1. Check JWT_SECRET is configured ─────────────────────
        if not JWT_SECRET:
            return jsonify({"error": "Server misconfiguration: JWT_PASSWORD not set."}), 500

        # ── 2. Extract token from Authorization header ────────────
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header. Use 'Bearer <token>'."}), 401

        token = auth_header.split("Bearer ", 1)[1].strip()
        if not token:
            return jsonify({"error": "Empty token provided."}), 401

        # ── 3. Decode and verify the JWT ──────────────────────────
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired."}), 401
        except jwt.InvalidTokenError as e:
            return jsonify({"error": f"Invalid token: {str(e)}"}), 401

        # ── 4. Extract and validate email ─────────────────────────
        email = payload.get("email")
        if not email:
            return jsonify({"error": "Token payload missing 'email' field."}), 400

        if not EMAIL_REGEX.match(email):
            return jsonify({"error": f"Invalid email format in token: {email}"}), 400

        # ── 5. Attach to Flask context ────────────────────────────
        g.jwt_payload = payload
        g.email = email

        return f(*args, **kwargs)

    return decorated
