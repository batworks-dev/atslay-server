# credits.py — Credit check, deduction, and audit logging for ATSlay
#
# Usage:
#   from credits import check_and_deduct_credit
#
#   ok, err = check_and_deduct_credit(g.email, "resume-optimizer")
#   if not ok:
#       return err   # already a Flask (jsonify, status_code) tuple

from datetime import datetime, timezone
from flask import jsonify
from db import get_users_collection, get_credit_usage_collection


# ─────────────────────────────────────────────────────────────────────────────
# SERVICE REGISTRY
# Maps internal service keys to human-readable display names and credit cost.
# ─────────────────────────────────────────────────────────────────────────────

SERVICES = {
    "resume-optimizer": {
        "display_name": "AI Resume Optimizer",
        "credits_cost": 1,
    },
    "score-resume-jd": {
        "display_name": "Resume vs JD Scorer",
        "credits_cost": 0.5,
    },
    "score-resume": {
        "display_name": "Resume Scorer",
        "credits_cost": 0,
    },
}


def check_and_deduct_credit(email: str, service_key: str):
    """
    1. Look up the user's credit balance in the 'users' collection.
    2. If credits == 0 (or user not found), return an HTTP 402 error tuple.
    3. Atomically decrement credits by the service cost using $inc.
    4. Write an audit row to 'credit-usage'.
    5. Return (True, None) on success.

    Returns:
        (True, None)                         — success, caller should continue
        (False, (jsonify_response, status))  — failure, caller should return this
    """
    service = SERVICES.get(service_key)
    if not service:
        # Unknown service key — don't block the request, just skip billing
        return True, None

    cost = service["credits_cost"]

    # ── Free service: log an audit entry and return immediately ──────────
    if cost == 0:
        usage_col = get_credit_usage_collection()
        try:
            usage_col.insert_one({
                "email":        email,
                "service":      service_key,
                "service_name": service["display_name"],
                "credits_used": 0,
                "timestamp":    datetime.now(timezone.utc),
            })
        except Exception as log_err:
            print(f"⚠️  Failed to write credit-usage log: {log_err}")
        return True, None
    users_col = get_users_collection()
    usage_col = get_credit_usage_collection()

    # ── 1. Fetch current credit balance ───────────────────────────────────
    user_doc = users_col.find_one({"email": email}, {"tokens": 1})

    if user_doc is None:
        return False, (
            jsonify({
                "error": "User account not found.",
                "code": "USER_NOT_FOUND",
            }),
            404,
        )

    current_credits = user_doc.get("tokens", 0)

    if current_credits < cost:
        return False, (
            jsonify({
                "error": "Insufficient credits. Please top up your account to use this service.",
                "code": "INSUFFICIENT_CREDITS",
                "credits_available": current_credits,
                "credits_required": cost,
            }),
            402,
        )

    # ── 2. Atomically deduct credits ──────────────────────────────────────
    result = users_col.update_one(
        {"email": email, "tokens": {"$gte": cost}},   # guard against race condition
        {"$inc": {"tokens": -cost}},
    )

    if result.modified_count == 0:
        # Race condition: another request consumed the last credit first
        return False, (
            jsonify({
                "error": "Insufficient credits. Please top up your account to use this service.",
                "code": "INSUFFICIENT_CREDITS",
                "credits_available": 0,
                "credits_required": cost,
            }),
            402,
        )

    # ── 3. Write audit log entry ──────────────────────────────────────────
    try:
        usage_col.insert_one({
            "email":        email,
            "service":      service_key,
            "service_name": service["display_name"],
            "credits_used": cost,
            "timestamp":    datetime.now(timezone.utc),
        })
    except Exception as log_err:
        # Never block the main request because of a logging failure
        print(f"⚠️  Failed to write credit-usage log: {log_err}")

    return True, None
