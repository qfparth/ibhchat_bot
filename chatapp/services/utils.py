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
    city__city__icontains       when city is mentioned
    category__name__icontains   when a business category is mentioned
    business_name__icontains    when a business name is mentioned
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

# ─────────────────────────────────────────────
# IBH PLATFORM KNOWLEDGE — for LLaMA guidance
# ─────────────────────────────────────────────
IBH_KNOWLEDGE = """
You are a smart assistant for Indian Business Hub (IBH) — indianbusinesshub.in

ABOUT IBH:
- Indian Business Hub is a FREE online business directory for Indian businesses
- It helps businesses get discovered by customers across India
- Businesses can list for FREE, get verified, and grow digitally

KEY FEATURES:
1. Free Business Listing — Any business can list on IBH for free
2. Verified Badge — Upload documents to get a verified badge for trust
3. Digital Visiting Card — Share business details on WhatsApp, Instagram
4. Festival & Brand Designs — Auto-create festival promo images with your logo
5. One-Page Website — Get a simple website for your business, no coding needed
6. Easy Customer Contact — Customers can call/message businesses directly
7. IBH Mobile App — Available FREE on Google Play

HOW TO ADD A BUSINESS:
Step 1: Go to indianbusinesshub.in and click 'Add Business'
Step 2: Enter business name, phone, email, address, city, category
Step 3: Set a password and click Finish
- It is completely FREE

HOW TO GET VERIFIED:
Step 1: Complete your business profile fully
Step 2: Upload ID proof, business certificate, GST document
Step 3: IBH team reviews and approves
Step 4: Get your Verified Badge
- Use the IBH App to upload documents from phone camera

IBH APP FEATURES:
- Festival & Brand Designs with your logo
- Digital Visiting Card
- One-Page Website
- Easy Customer Contact
- Verification via phone camera
- Download: play.google.com/store/apps/details?id=com.app.indianbusinesshub

HOW TO SEARCH FOR A BUSINESS:
- Type the business type and city: example 'restaurants in Ahmedabad'
- Or type what you need: example 'I need a tailor in Surat'
- Browse by category and city on the website

CONTACT IBH:
- Email: info.indianbusinesshub@gmail.com
- Phone: +91 8000841620
- Address: 209-A, Satva Icon, Vastral, Ahmedabad, Gujarat 382418

TESTIMONIALS:
- Raigo Ceramica: Got inquiries within a week of verifying
- Accufix: Got bulk order calls for precision tools
- U Smile: Toy store got more orders after listing

TIPS FOR BUSINESS OWNERS ON IBH:
- Complete your profile 100% for 3x more views
- Add services in detail so customers find you by keywords
- Share your IBH profile on WhatsApp and Instagram
- Use festival designs from the app on social media
- Get verified to build trust with customers
"""

# ─────────────────────────────────────────────
# GUIDANCE TRIGGER KEYWORDS
# ─────────────────────────────────────────────
GUIDANCE_TRIGGERS = [
    # platform questions
    "what is ibh", "about ibh", "what is indian business hub",
    "what is this website", "what does ibh do", "tell me about ibh",
    "what is indianbusinesshub",
    # listing
    "how to add", "add my business", "list my business",
    "how to list", "how to register", "get listed", "join ibh",
    "add business", "list business", "register business",
    # verification
    "how to get verified", "get verified", "verified badge",
    "verify my business", "verification process", "verify business",
    # app
    "ibh app", "download app", "install app", "app features",
    "mobile app", "what does the app", "app benefits",
    # benefits
    "why use ibh", "benefits of ibh", "is ibh free", "free listing",
    "why list on ibh", "advantages of ibh", "why ibh",
    # contact
    "contact ibh", "ibh contact", "ibh phone", "ibh email",
    "ibh address", "ibh location", "ibh support",
    # new user
    "i am new", "new user", "how to use ibh", "getting started",
    "first time", "where do i start", "i just joined",
    "help me get started", "how to use this",
    # search help
    "how to search", "how to find business", "how to find",
    "how do i search", "how to look for",
    # tips
    "tips for ibh", "how to grow on ibh", "get more customers ibh",
    "how to get more views", "improve my listing",
]


def is_guidance_question(user_msg: str) -> bool:
    """Check if user is asking about IBH platform guidance."""
    msg = user_msg.lower().strip()
    return any(trigger in msg for trigger in GUIDANCE_TRIGGERS)


# ─────────────────────────────────────────────
# DYNAMIC GUIDANCE HANDLER — LLaMA3 powered
# ─────────────────────────────────────────────
def handle_guidance(user_msg: str) -> str:
    """
    Uses LLaMA3 to dynamically answer any IBH platform guidance question.
    LLaMA3 reads the IBH knowledge base and gives accurate, friendly answers.
    """
    system = (
        "You are a helpful and friendly assistant for Indian Business Hub (IBH). "
        "You guide users on how to use the IBH platform accurately. "
        "Use the IBH knowledge provided to answer. "
        "Keep answers clear, friendly, and use emojis where appropriate. "
        "Never make up information not in the knowledge base. "
        "If the question is not in the knowledge base, say you are not sure and suggest contacting IBH support."
    )

    prompt = f"""
{IBH_KNOWLEDGE}

User Question: {user_msg}

Answer the user's question accurately based on the IBH knowledge above.
Be friendly, use emojis, keep it clear and helpful.

Answer:
"""
    try:
        return ask_llama(prompt, system=system).strip()
    except Exception as e:
        logger.error("handle_guidance failed: %s", e)
        return (
            "I'm here to help with Indian Business Hub! 😊\n\n"
            "For any questions, contact us:\n"
            "📧 info.indianbusinesshub@gmail.com\n"
            "📞 +91 8000841620"
        )


# ─────────────────────────────────────────────
# FAST INTENT DETECTION
# ─────────────────────────────────────────────
def fast_intent(user_msg: str) -> str | None:
    msg = user_msg.lower().strip()
    msg_clean = msg.strip("?!., ")

    if msg_clean in GREETING_WORDS or msg in GREETING_WORDS:
        return "greeting"
    if msg_clean in FAREWELL_WORDS or msg in FAREWELL_WORDS:
        return "farewell"
    if msg_clean in THANKS_WORDS or msg in THANKS_WORDS:
        return "thanks"
    if msg_clean in HELP_WORDS or msg in HELP_WORDS:
        return "help"

    greeting_triggers = [
        "hi", "hello", "hey", "good morning", "good afternoon",
        "good evening", "good night", "namaste", "namaskar", "greetings"
    ]
    farewell_triggers = ["bye", "goodbye", "see you", "take care", "farewell"]
    thanks_triggers   = ["thank", "thanks", "thx", "appreciated", "dhanyawad"]
    help_triggers     = ["what can you do", "how to use", "what do you do"]

    for trigger in greeting_triggers:
        if msg_clean == trigger or msg_clean.startswith(trigger + " ") or trigger in msg_clean:
            return "greeting"

    for trigger in farewell_triggers:
        if msg_clean == trigger or msg_clean.startswith(trigger):
            return "farewell"

    for trigger in thanks_triggers:
        if trigger in msg_clean:
            return "thanks"

    for trigger in help_triggers:
        if trigger in msg_clean:
            return "help"

    # ── Guidance check — BEFORE general ──
    if is_guidance_question(user_msg):
        return "guidance"

    # ── General knowledge check — BEFORE LLaMA ──
    for trigger in GENERAL_TRIGGERS:
        if msg_clean.startswith(trigger) or f" {trigger} " in f" {msg_clean} ":
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
You are an intent classifier for a business directory chatbot called Indian Business Hub (IBH).

Classify the user message into exactly one of these intents:
- "greeting"  -> hi, hello, good morning, hey, etc.
- "farewell"  -> bye, goodbye, see you, take care, etc.
- "thanks"    -> thank you, thanks, thx, appreciated, etc.
- "help"      -> what can you do, how to use, help me, etc.
- "guidance"  -> questions about IBH platform: how to add business, verification, app features, benefits, contact IBH, how to use the website, tips for IBH
- "search"    -> looking for a business, service, category, city, visits, stats in the directory
- "general"   -> asking for advice, tips, how-to, suggestions, general business knowledge
- "unknown"   -> anything else

Examples:
Message: "how to improve my IT business" -> general
Message: "tips for growing my restaurant" -> general
Message: "show restaurants in Ahmedabad" -> search
Message: "hi" -> greeting
Message: "how to add my business on IBH" -> guidance
Message: "how to get verified" -> guidance
Message: "what does the IBH app do" -> guidance
Message: "is IBH free" -> guidance
Message: "what is Indian Business Hub" -> guidance
Message: "how to search for a business" -> guidance
Message: "how to get more customers on IBH" -> guidance

Return ONLY the intent label as a single lowercase word.

User Message: {user_msg}
Intent:
"""
    try:
        response = ask_llama(prompt)
        intent = response.strip().lower().split()[0].strip(".,!?")
        if intent not in {"greeting", "farewell", "thanks", "help", "guidance", "search", "general", "unknown"}:
            return "unknown"
        return intent
    except Exception as e:
        logger.error("detect_intent failed: %s", e)
        return "unknown"


# ─────────────────────────────────────────────
# CONVERSATIONAL HANDLER
# ─────────────────────────────────────────────
def handle_conversational(intent: str, user_msg: str) -> str:
    COMPANY_NAME = "Indian Business Hub"

    system = (
        f"You are a friendly assistant for {COMPANY_NAME} (IBH). "
        "You help users find businesses and services in their city. "
        "Always use emojis to keep replies warm and friendly. "
        "Keep replies SHORT (2-3 sentences) and conversational. "
        "NEVER mention any business data, database, or listings in your reply."
    )

    contexts = {
        "greeting": (
            f"The user greeted you. Reply warmly with emojis. "
            f"Start with: 'Hello! Welcome to {COMPANY_NAME}! 😊' "
            f"Then offer to help — find businesses OR guide them on using IBH. "
            f"Example: 'I can help you find businesses or guide you on how to use IBH!'"
        ),
        "farewell": "User is leaving. Wish them well with emojis and invite them back.",
        "thanks":   "User thanked you. Accept graciously with emojis and offer further help.",
        "help": (
            f"Explain what you can do for {COMPANY_NAME}: "
            "1. Find businesses by city and category "
            "2. Guide users on how to use IBH platform "
            "3. Help business owners list and verify their business "
            "Use emojis. Keep it short and friendly."
        ),
        "unknown": (
            "Respond warmly with emojis. "
            "Ask if they want to find a business OR need help using the IBH platform."
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
        return f"Hello! Welcome to {COMPANY_NAME}! 😊 I can help you find businesses or guide you on using IBH. Just ask!"


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

    matched = GENERAL_ADVICE["default"]
    for category, data in GENERAL_ADVICE.items():
        if category == "default":
            continue
        if any(kw in msg for kw in data["keywords"]):
            matched = data
            break

    tips = matched["tips"]
    response = "Here are 5 great tips to help you:\n\n"
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
      1. Detect intent
      2. guidance   → LLaMA3 dynamically answers IBH platform questions
      3. greeting / farewell / thanks / help → conversational
      4. general    → template tips
      5. search     → SQL → ORM → semantic → LLaMA final answer
      6. unknown    → fallback
    """
    user_msg = user_msg.strip()
    if not user_msg:
        return "I didn't catch that. Could you please type your question?"

    intent = detect_intent(user_msg)
    logger.info("Intent: %s | Message: %s", intent, user_msg)
    print(f"[CHAT] Intent: {intent} | Message: {user_msg}")

    # ── Dynamic IBH guidance — LLaMA3 powered ──
    if intent == "guidance":
        return handle_guidance(user_msg)

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

    # ── Fallback 3: No DB results → guidance check ──
    if not db_context.strip():
        if is_guidance_question(user_msg):
            return handle_guidance(user_msg)
        logger.info("No DB results found, falling back to general answer.")
        return handle_general(user_msg)

    final_prompt = f"""
You are a friendly and knowledgeable assistant for Indian Business Hub (IBH).

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