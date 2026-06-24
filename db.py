# db.py — MongoDB connection for ATSlay

import os
from pymongo import MongoClient, ASCENDING, DESCENDING
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
# MONGODB CLIENT
# ─────────────────────────────────────────────

_client = None
_db = None


def _get_db():
    """Lazy-initialize the MongoDB client and return the database."""
    global _client, _db
    if _db is None:
        uri = os.environ.get("DB_URI")
        if not uri:
            raise RuntimeError("DB_URI is not set in environment variables.")
        _client = MongoClient(uri)
        _db = _client.get_default_database()
    return _db


def get_collection():
    """Return the 'ats-scoring' collection."""
    db = _get_db()
    return db["ats-scoring"]


def ensure_indexes():
    """Create recommended indexes on the ats-scoring collection."""
    collection = get_collection()
    collection.create_index([("email", ASCENDING)])
    collection.create_index([("processedAt", DESCENDING)])
    collection.create_index([("email", ASCENDING), ("type", ASCENDING)])
    print("✅ MongoDB indexes ensured on 'ats-scoring' collection.")


# Run index creation on import
try:
    ensure_indexes()
except Exception as e:
    print(f"⚠️  Could not ensure indexes on startup: {e}")
