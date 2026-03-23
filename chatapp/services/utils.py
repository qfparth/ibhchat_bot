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
# Text2SQL: LLaMA generates SQL
# ─────────────────────────────────────────────
def generate_sql(user_msg: str) -> str:
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
        response = re.sub(r"```sql|```", "", response).strip()
        if response.upper() == "NONE" or not response:
            return ""
        return response
    except Exception as e:
        logger.error("generate_sql failed: %s", e)
        return ""


# ─────────────────────────────────────────────
# Execute SQL safely
# ─────────────────────────────────────────────
def run_sql(sql: str) -> str:
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

        lines = []
        for row in rows:
            parts = [f"{col}: {val}" for col, val in zip(columns, row) if val is not None]
            lines.append(" | ".join(parts))

        return "\n".join(lines)

    except Exception as e:
        logger.error("run_sql failed: %s | SQL: %s", e, sql)
        return ""


# ─────────────────────────────────────────────
# ORM fallback (simple filters)
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
# Semantic search (lazy-loaded cache)
# ─────────────────────────────────────────────
def _load_embeddings():
    global DATA_LIST, DATA_EMBEDDINGS

    users = (
        User.objects
        .select_related("category", "city")
        .filter(is_active=1)
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
# FAST KEYWORD MATCHER — runs before LLaMA
# ─────────────────────────────────────────────
GREETING_WORDS = {
    "hi", "hello", "hey", "helo", "hii", "hiii",
    "good morning", "good afternoon", "good evening",
    "good night", "howdy", "greetings", "sup", "what's up",
    "whats up", "namaste", "namaskar", "hi there", "hello there",
    "hi good morning", "hello good morning", "hey good morning",
    "good morning everyone", "good evening everyone",
}

FAREWELL_WORDS = {
    "bye", "goodbye", "good bye", "see you", "see ya",
    "take care", "later", "cya", "ttyl", "farewell"
}

THANKS_WORDS = {
    "thanks", "thank you", "thank you so much", "thx",
    "ty", "appreciated", "cheers", "dhanyawad", "shukriya"
}

HELP_WORDS = {
    "help", "what can you do", "how to use", "what is this",
    "how does this work", "guide me", "assist me",
    "what do you do", "how can you help"
}

GENERAL_TRIGGERS = [
    "how to", "how do i", "how can i", "how should i",
    "tips for", "tips on", "advice", "suggest", "suggestion",
    "best way", "best ways", "best practice", "best strategies",
    "what is", "what are", "explain", "tell me about",
    "improve", "grow", "increase", "boost", "help me with",
    "why is", "why do", "benefits of", "difference between",
    "how does", "what does", "guide", "strategy", "strategies",
    "marketing", "digital marketing", "social media",
    "get more customers", "attract customers",
]


def fast_intent(user_msg: str) -> str | None:
    msg = user_msg.lower().strip()

    # ── Exact match first ──
    if msg in GREETING_WORDS:
        return "greeting"
    if msg in FAREWELL_WORDS:
        return "farewell"
    if msg in THANKS_WORDS:
        return "thanks"
    if msg in HELP_WORDS:
        return "help"

    # ── Partial / starts-with match ──
    greeting_triggers = [
        "hi", "hello", "hey", "good morning", "good afternoon",
        "good evening", "good night", "namaste", "namaskar", "greetings"
    ]
    farewell_triggers = ["bye", "goodbye", "see you", "take care", "farewell"]
    thanks_triggers   = ["thank", "thanks", "thx", "appreciated", "dhanyawad"]
    help_triggers     = ["what can you do", "how to use", "what do you do"]

    for trigger in greeting_triggers:
        if msg == trigger or msg.startswith(trigger + " ") or trigger in msg:
            return "greeting"

    for trigger in farewell_triggers:
        if msg == trigger or msg.startswith(trigger):
            return "farewell"

    for trigger in thanks_triggers:
        if trigger in msg:
            return "thanks"

    for trigger in help_triggers:
        if trigger in msg:
            return "help"

    # ── General knowledge check — BEFORE LLaMA ──
    for trigger in GENERAL_TRIGGERS:
        if msg.startswith(trigger) or f" {trigger} " in f" {msg} ":
            return "general"

    return None


# ─────────────────────────────────────────────
# INTENT DETECTION — keyword first, LLaMA second
# ─────────────────────────────────────────────
def detect_intent(user_msg: str) -> str:
    quick = fast_intent(user_msg)
    if quick:
        logger.info("Fast intent matched: %s", quick)
        return quick

    prompt = f"""
You are an intent classifier for a business directory chatbot.

Classify the user message into exactly one of these intents:
- "greeting"  → hi, hello, good morning, hey, etc.
- "farewell"  → bye, goodbye, see you, take care, etc.
- "thanks"    → thank you, thanks, thx, appreciated, etc.
- "help"      → what can you do, how to use, help me, etc.
- "search"    → looking for a business, service, category, city, visits, stats in the directory
- "general"   → asking for advice, tips, how-to, suggestions, general knowledge questions
- "unknown"   → anything else

Examples:
Message: "how to improve my IT business" → general
Message: "tips for growing my restaurant" → general
Message: "best marketing strategies" → general
Message: "show restaurants in Ahmedabad" → search
Message: "top 5 visited businesses" → search
Message: "hi" → greeting
Message: "what can you do" → help

Return ONLY the intent label as a single lowercase word.

User Message: {user_msg}
Intent:
"""
    try:
        response = ask_llama(prompt)
        intent = response.strip().lower().split()[0].strip(".,!?")
        if intent not in {"greeting", "farewell", "thanks", "help", "search", "general", "unknown"}:
            return "unknown"
        return intent
    except Exception as e:
        logger.error("detect_intent failed: %s", e)
        return "unknown"


# ─────────────────────────────────────────────
# CONVERSATIONAL HANDLER — updated with system prompt
# ─────────────────────────────────────────────
def handle_conversational(intent: str, user_msg: str) -> str:
    # ── System prompt — tells LLaMA who it is ──
    system = (
        "You are a friendly assistant for a business directory platform. "
        "You help users find businesses and services in their city. "
        "Keep replies SHORT (2-3 sentences), warm, and conversational. "
        "NEVER mention any business data, database, or listings in your reply."
    )

    contexts = {
        "greeting": (
            "The user greeted you. Reply warmly. "
            "Say hello and offer to help find businesses or services."
        ),
        "farewell": "User is leaving. Wish them well and invite them back.",
        "thanks":   "User thanked you. Accept graciously and offer further help.",
        "help": (
            "Explain what you can do: find businesses by city/category, "
            "search services, show top visited businesses. "
            "Examples: 'restaurants in Ahmedabad', 'IT services in Surat'."
        ),
        "unknown": (
            "Respond warmly and guide them to search for a business. "
            "Give 1-2 example searches they can try."
        ),
    }

    prompt = f"""
Situation: {contexts.get(intent, contexts['unknown'])}
User Message: "{user_msg}"
Your Reply:
"""
    try:
        return ask_llama(prompt, system=system).strip()
    except Exception as e:
        logger.error("handle_conversational failed: %s", e)
        return "Hello! I'm here to help you find businesses and services. Try: 'restaurants in Ahmedabad'."


# ─────────────────────────────────────────────
# GENERAL KNOWLEDGE HANDLER — updated with system prompt
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# GENERAL KNOWLEDGE HANDLER — template based
# ─────────────────────────────────────────────

GENERAL_ADVICE = {
    "it": {
        "keywords": ["it", "software", "tech", "technology", "computer", "digital", "app", "website"],
        "tips": [
            "Build a strong online presence with a professional website and active social media.",
            "Focus on client retention through excellent after-sales support and follow-ups.",
            "Stay updated with the latest technologies, certifications, and industry trends.",
            "Collect and showcase customer reviews, case studies, and success stories.",
            "Network with other businesses, attend tech events, and explore partnerships.",
        ]
    },
    "restaurant": {
        "keywords": ["restaurant", "food", "cafe", "catering", "hotel", "dining", "eat"],
        "tips": [
            "Maintain consistent food quality and hygiene standards at all times.",
            "Use social media to showcase your dishes with attractive photos and videos.",
            "Offer loyalty programs, discounts, and special deals to retain customers.",
            "Collect customer feedback and act on it to improve your menu and service.",
            "Partner with food delivery platforms to reach a wider audience.",
        ]
    },
    "retail": {
        "keywords": ["shop", "store", "retail", "apparel", "clothing", "fashion", "sell"],
        "tips": [
            "Keep your inventory updated with trending products and seasonal items.",
            "Create an engaging storefront both physically and online.",
            "Run promotions, sales, and loyalty programs to attract repeat customers.",
            "Use social media marketing to showcase products and reach new buyers.",
            "Provide excellent customer service to build trust and word-of-mouth referrals.",
        ]
    },
    "marketing": {
        "keywords": ["marketing", "advertise", "promotion", "brand", "social media", "digital marketing"],
        "tips": [
            "Define your target audience clearly before planning any marketing campaign.",
            "Use a mix of social media, email, and content marketing for best results.",
            "Create valuable content that educates and engages your potential customers.",
            "Track your marketing metrics regularly and optimize based on what works.",
            "Invest in local SEO so customers in your city can find you easily online.",
        ]
    },
    "customer": {
        "keywords": ["customer", "client", "visitors", "attract", "more customers", "get customers"],
        "tips": [
            "Provide exceptional service that makes customers want to return and refer others.",
            "Ask satisfied customers for reviews and testimonials on Google and social media.",
            "Run referral programs that reward customers for bringing new clients.",
            "Stay active on social media and engage with your audience regularly.",
            "Offer first-time discounts or free trials to attract new customers.",
        ]
    },
    "default": {
        "keywords": [],
        "tips": [
            "Build a strong online presence with a professional website and social media profiles.",
            "Focus on delivering exceptional customer service to build loyalty and referrals.",
            "Invest in digital marketing — social media, email, and local SEO.",
            "Keep your skills, products, and services updated with market trends.",
            "Collect customer reviews and use feedback to continuously improve.",
        ]
    }
}


def handle_general(user_msg: str) -> str:
    """
    Answer general business questions using smart templates.
    No LLaMA call — avoids DB context contamination entirely.
    """
    msg = user_msg.lower()

    # ── Match best category ──
    matched = GENERAL_ADVICE["default"]
    for category, data in GENERAL_ADVICE.items():
        if category == "default":
            continue
        if any(kw in msg for kw in data["keywords"]):
            matched = data
            break

    # ── Build response ──
    tips = matched["tips"]
    response = f"Here are 5 great tips to help you:\n\n"
    for i, tip in enumerate(tips, 1):
        response += f"{i}. {tip}\n"
    response += "\nYou can also explore related businesses in your city on our platform for more inspiration!"

    return response.strip()


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

    # ── Conversational intents ──
    if intent in {"greeting", "farewell", "thanks", "help"}:
        return handle_conversational(intent, user_msg)

    # ── General knowledge / advice ──
    if intent == "general":
        return handle_general(user_msg)

    # ── Short/unclear messages ──
    if intent == "unknown":
        if len(user_msg.split()) <= 4:
            return handle_conversational("greeting", user_msg)
        return handle_general(user_msg)

    # ── Search: Text2SQL first ──
    db_context = ""

    sql = generate_sql(user_msg)
    if sql:
        logger.info("Generated SQL: %s", sql)
        db_context = run_sql(sql)

    # ── Fallback 1: ORM ──
    if not db_context.strip():
        logger.info("SQL returned nothing, trying ORM filter.")
        db_context = get_dynamic_data(user_msg)

    # ── Fallback 2: Semantic search ──
    if not db_context.strip():
        logger.info("ORM returned nothing, trying semantic search.")
        db_context = semantic_search(user_msg)

    # ── Fallback 3: No DB results → treat as general question ──
    if not db_context.strip():
        logger.info("No DB results found, falling back to general answer.")
        return handle_general(user_msg)

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