from sentence_transformers import SentenceTransformer, util
from chatapp.models import User, Category, Service, ProfileDailyVisit
from chatapp.services.ai import ask_llama
from django.db import connection

import json
import re
import logging

logger = logging.getLogger(__name__)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Lazy-loaded globals
DATA_LIST = None
DATA_EMBEDDINGS = None

# ─────────────────────────────────────────────
# DATABASE SCHEMA CONTEXT FOR LLAMA
# ─────────────────────────────────────────────
DB_SCHEMA = """
You have access to a MySQL database with these tables:

1. users
   - id, name, email, phone, business_name
   - category_id (FK → categories.id)
   - city (FK → cities.id)
   - state (int), pincode, address
   - description, website
   - latitude, longitude
   - is_active (1=active), is_verified (1=verified)
   - created_at, deleted_at

2. categories
   - id, name, description, keywords
   - is_active (1=active)
   - category_type: enum('festival','daily','greeting','motivation','business')

3. cities
   - id, city (name), state_id, is_top (1=top city)

4. services
   - id, service_title, description, keywords
   - user_id (FK → users.id)
   - category_id (FK → categories.id)
   - is_active (1=active)
   - deleted_at (NULL = not deleted)

5. profile_daily_visits
   - id, profile_id (FK → users.id)
   - visit_date (DATE), visits (int)

KEY RULES for joining:
- users.city = cities.id
- users.category_id = categories.id
- services.user_id = users.id
- services.category_id = categories.id
- profile_daily_visits.profile_id = users.id
- Always filter: users.deleted_at IS NULL, services.deleted_at IS NULL
- Always filter: users.is_active = 1 for active businesses
"""

# ─────────────────────────────────────────────
# ALLOWED FIELDS (security whitelist — ORM path)
# ─────────────────────────────────────────────
ALLOWED_FILTER_FIELDS = {
    "city__city__icontains",
    "category__name__icontains",
    "business_name__icontains",
}

# ─────────────────────────────────────────────
# SQL SAFETY — whitelist allowed tables/keywords
# ─────────────────────────────────────────────
ALLOWED_TABLES = {"users", "categories", "cities", "services", "profile_daily_visits"}
FORBIDDEN_KEYWORDS = {"drop", "delete", "truncate", "insert", "update", "alter", "create", "grant", "revoke"}


def is_safe_sql(sql: str) -> bool:
    """Block any destructive SQL — only SELECT allowed."""
    sql_lower = sql.lower()
    if not sql_lower.strip().startswith("select"):
        return False
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", sql_lower):
            return False
    return True


# ─────────────────────────────────────────────
# STEP 1 — Text2SQL: LLaMA generates SQL
# ─────────────────────────────────────────────
def generate_sql(user_msg: str) -> str:
    """
    Ask LLaMA to convert natural language question → safe MySQL SELECT query.
    Returns empty string on failure.
    """
    prompt = f"""
{DB_SCHEMA}

STRICT RULES:
- Return ONLY a valid MySQL SELECT query — no explanation, no backticks, no markdown.
- Always use table aliases for clarity (e.g. u for users, c for cities).
- Always include LIMIT (default 20, max 100).
- Never use DROP, DELETE, UPDATE, INSERT, ALTER.
- For business searches always filter: u.deleted_at IS NULL AND u.is_active = 1
- For service searches always filter: s.deleted_at IS NULL AND s.is_active = 1
- Use LIKE with % for text searches (e.g. c.name LIKE '%ahmedabad%').
- If question is unclear or not related to the DB, return only: NONE

Examples:

User: "show restaurants in Ahmedabad"
SQL: SELECT u.business_name, u.phone, u.address, cat.name as category, ci.city
     FROM users u
     JOIN categories cat ON u.category_id = cat.id
     JOIN cities ci ON u.city = ci.id
     WHERE ci.city LIKE '%Ahmedabad%'
     AND cat.name LIKE '%restaurant%'
     AND u.deleted_at IS NULL AND u.is_active = 1
     LIMIT 20;

User: "top 5 most visited businesses"
SQL: SELECT u.business_name, u.phone, SUM(p.visits) as total_visits
     FROM profile_daily_visits p
     JOIN users u ON p.profile_id = u.id
     WHERE u.deleted_at IS NULL AND u.is_active = 1
     GROUP BY p.profile_id, u.business_name, u.phone
     ORDER BY total_visits DESC
     LIMIT 5;

User: "what services are available for IT category"
SQL: SELECT s.service_title, s.description, u.business_name, u.phone
     FROM services s
     JOIN users u ON s.user_id = u.id
     JOIN categories cat ON s.category_id = cat.id
     WHERE cat.name LIKE '%IT%'
     AND s.deleted_at IS NULL AND s.is_active = 1
     LIMIT 20;

User Question: {user_msg}
SQL:
"""
    try:
        response = ask_llama(prompt).strip()
        # Clean up any accidental markdown
        response = re.sub(r"```sql|```", "", response).strip()
        if response.upper() == "NONE" or not response:
            return ""
        return response
    except Exception as e:
        logger.error("generate_sql failed: %s", e)
        return ""


# ─────────────────────────────────────────────
# STEP 2 — Execute SQL safely
# ─────────────────────────────────────────────
def run_sql(sql: str) -> str:
    """
    Execute a SELECT query and return results as a formatted string.
    Returns empty string if unsafe or no results.
    """
    if not sql or not is_safe_sql(sql):
        logger.warning("Blocked unsafe SQL: %s", sql)
        return ""

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()

        if not rows:
            return ""

        # Format as readable text for LLaMA context
        lines = []
        for row in rows:
            parts = [f"{col}: {val}" for col, val in zip(columns, row) if val is not None]
            lines.append(" | ".join(parts))

        return "\n".join(lines)

    except Exception as e:
        logger.error("run_sql failed: %s | SQL: %s", e, sql)
        return ""


# ─────────────────────────────────────────────
# STEP 3 — ORM fallback (simple filters)
# ─────────────────────────────────────────────
def generate_query(user_msg: str) -> dict:
    prompt = f"""
You are a Django ORM expert.

Convert the user question into a JSON object with Django ORM filter keys.

STRICT RULES:
- Return ONLY valid JSON — no explanation, no backticks, no markdown.
- Allowed filter keys:
    city__city__icontains       → when city is mentioned
    category__name__icontains   → when a business category is mentioned
    business_name__icontains    → when a business name is mentioned
- Also add a "limit" key (integer) if the user asks for a specific count.
- If nothing matches, return: {{}}

User Question: {user_msg}
"""
    try:
        response = ask_llama(prompt)
        return json.loads(response)
    except json.JSONDecodeError:
        logger.warning("LLaMA returned non-JSON for query generation: %s", response)
        return {}
    except Exception as e:
        logger.error("generate_query failed: %s", e)
        return {}


def get_dynamic_data(user_msg: str) -> str:
    """ORM-based fallback for simple user/category/city searches."""
    query_dict = generate_query(user_msg)

    try:
        limit = int(query_dict.get("limit", 20))
        limit = max(1, min(limit, 100))
    except (TypeError, ValueError):
        limit = 20

    safe_filter = {
        key: value
        for key, value in query_dict.items()
        if key in ALLOWED_FILTER_FIELDS
    }

    try:
        users = (
            User.objects
            .select_related("category", "city")
            .filter(
                **safe_filter,
                category__isnull=False,
                city__isnull=False,
                is_active=1,
                deleted_at__isnull=True,
            )[:limit]
        )
    except Exception as e:
        logger.error("ORM filter failed: %s | Error: %s", safe_filter, e)
        return ""

    if not users:
        return ""

    lines = []
    for u in users:
        category_name = u.category.name if u.category_id else "Unknown"
        city_name = u.city.city if u.city_id else "Unknown"
        lines.append(
            f"Business: {u.business_name} | Category: {category_name} | "
            f"City: {city_name} | Phone: {u.phone}"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────
# STEP 4 — Semantic search (lazy-loaded cache)
# ─────────────────────────────────────────────
def _load_embeddings():
    global DATA_LIST, DATA_EMBEDDINGS

    users = (
        User.objects
        .select_related("category", "city")
        .filter(is_active=1, deleted_at__isnull=True)
    )
    texts = []
    for u in users:
        category_name = u.category.name if u.category_id else "Unknown"
        city_name = u.city.city if u.city_id else "Unknown"
        texts.append(
            f"{u.business_name} is a {category_name} service in {city_name}"
        )

    DATA_LIST = texts
    DATA_EMBEDDINGS = model.encode(texts, convert_to_tensor=True)
    logger.info("Loaded %d businesses into semantic index.", len(texts))


def semantic_search(user_msg: str, top_k: int = 10) -> str:
    global DATA_LIST, DATA_EMBEDDINGS

    if DATA_LIST is None:
        _load_embeddings()

    if not DATA_LIST:
        return ""

    query_embedding = model.encode(user_msg.lower(), convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, DATA_EMBEDDINGS)[0]
    top_indices = scores.argsort(descending=True)[:top_k]

    return "\n".join(DATA_LIST[int(i)] for i in top_indices)


def reload_embeddings():
    global DATA_LIST, DATA_EMBEDDINGS
    DATA_LIST = None
    DATA_EMBEDDINGS = None
    _load_embeddings()


# ─────────────────────────────────────────────
# INTENT DETECTION
# ─────────────────────────────────────────────
def detect_intent(user_msg: str) -> str:
    """
    Returns: 'greeting' | 'farewell' | 'thanks' | 'help' | 'search' | 'unknown'
    """
    prompt = f"""
You are an intent classifier for a business directory chatbot.

Classify the user message into exactly one of these intents:
- "greeting"  → hi, hello, good morning, hey, etc.
- "farewell"  → bye, goodbye, see you, take care, etc.
- "thanks"    → thank you, thanks, thx, appreciated, etc.
- "help"      → what can you do, how to use, help me, etc.
- "search"    → looking for a business, service, category, city, visits, stats
- "unknown"   → anything else

Return ONLY the intent label as a single lowercase word.

User Message: {user_msg}
Intent:
"""
    try:
        response = ask_llama(prompt)
        intent = response.strip().lower().strip(".")
        if intent not in {"greeting", "farewell", "thanks", "help", "search", "unknown"}:
            return "unknown"
        return intent
    except Exception as e:
        logger.error("detect_intent failed: %s", e)
        return "unknown"


# ─────────────────────────────────────────────
# CONVERSATIONAL HANDLER
# ─────────────────────────────────────────────
def handle_conversational(intent: str, user_msg: str) -> str:
    persona = """
You are a friendly assistant for a business directory platform.
Help users find businesses, services, and companies in their city.
Keep replies SHORT (2-3 sentences), warm, and conversational.
Always end with a nudge to search for a business or service.
"""
    contexts = {
        "greeting": "User greeted you. Greet back warmly and offer to help find businesses or services.",
        "farewell": "User is leaving. Wish them well and invite them back anytime.",
        "thanks":   "User thanked you. Accept graciously and offer to help further.",
        "help": (
            "Explain what you can do: find businesses by city/category, "
            "search services, show top visited businesses, and more. "
            "Give examples: 'restaurants in Ahmedabad', "
            "'top 5 most visited businesses', 'IT services in Surat'."
        ),
        "unknown": "Respond helpfully and guide them to search for a business or service.",
    }

    prompt = f"""
{persona}

Situation: {contexts.get(intent, contexts['unknown'])}

User Message: "{user_msg}"

Your Reply:
"""
    try:
        return ask_llama(prompt).strip()
    except Exception as e:
        logger.error("handle_conversational failed: %s", e)
        return "Hi! I can help you find businesses and services. Try: 'restaurants in Ahmedabad'."


# ─────────────────────────────────────────────
# MAIN CHAT ENTRY POINT
# ─────────────────────────────────────────────
def chat(user_msg: str) -> str:
    """
    Full pipeline:
      1. Detect intent → handle conversational if not search
      2. Text2SQL → run SQL → get DB context  (primary)
      3. ORM filter fallback                   (secondary)
      4. Semantic search fallback              (tertiary)
      5. Feed context to LLaMA → final answer
    """
    user_msg = user_msg.strip()
    if not user_msg:
        return "I didn't catch that. Could you please type your question?"

    intent = detect_intent(user_msg)
    logger.info("Intent: %s | Message: %s", intent, user_msg)

    # ── Non-search: handle conversationally ──
    if intent in {"greeting", "farewell", "thanks", "help", "unknown"}:
        return handle_conversational(intent, user_msg)

    # ── Search: try Text2SQL first ──
    db_context = ""

    sql = generate_sql(user_msg)
    if sql:
        logger.info("Generated SQL: %s", sql)
        db_context = run_sql(sql)

    # ── Fallback 1: ORM filter ──
    if not db_context.strip():
        logger.info("SQL returned nothing, trying ORM filter.")
        db_context = get_dynamic_data(user_msg)

    # ── Fallback 2: Semantic search ──
    if not db_context.strip():
        logger.info("ORM returned nothing, trying semantic search.")
        db_context = semantic_search(user_msg)

    if not db_context.strip():
        db_context = "No relevant businesses or services found in the database."

    final_prompt = f"""
You are a friendly and knowledgeable business directory assistant.

Use ONLY the data below to answer the user's question.
- Present results clearly and in a readable format.
- List multiple results with numbering if more than one.
- Include phone numbers when available.
- If data is not found, say: "I couldn't find that — try a different search term or city."
- End with a helpful follow-up suggestion.

Data:
{db_context}

User Question: {user_msg}

Answer:
"""
    try:
        return ask_llama(final_prompt).strip()
    except Exception as e:
        logger.error("Final LLaMA call failed: %s", e)
        return "Something went wrong. Please try again."


# ─────────────────────────────────────────────
# PAGINATION HELPER
# ─────────────────────────────────────────────
def get_categories(page: int = 1, limit: int = 10) -> dict:
    start = (page - 1) * limit
    end = start + limit
    total = Category.objects.count()
    names = list(Category.objects.values_list("name", flat=True)[start:end])
    return {"total": total, "page": page, "categories": names}