from sentence_transformers import SentenceTransformer, util
from chatapp.models import User
from chatapp.models import Category
from chatapp.services.ai import ask_llama

import json

model = SentenceTransformer("all-MiniLM-L6-v2")




def generate_query(user_msg):

    prompt = f"""
You are a Django ORM expert.

Convert user question into Django ORM filter in JSON format.

STRICT RULES:
- Return ONLY valid JSON
- No explanation
- No text
- No backticks
- Keys must match Django ORM

Example:
{{
  "city__city__icontains": "ahmedabad",
  "category__name__icontains": "apparel"
}}

User Question: {user_msg}
"""

    response = ask_llama(prompt)

    try:
        return json.loads(response)
    except:
        print("JSON Error:", response)
        return {}



def get_dynamic_data(user_msg):

    query_dict = generate_query(user_msg)

    if not query_dict:
        return ""

    # 🔒 allowed fields (security)
    allowed_fields = [
        "city__city__icontains",
        "category__name__icontains",
        "business_name__icontains"
    ]

    safe_query = {}

    for key, value in query_dict.items():
        if key in allowed_fields:
            safe_query[key] = value

    try:
        users = User.objects.select_related("category", "city").filter(
            **safe_query,
            category__isnull=False,
            city__isnull=False
        )[:20]

    except Exception as e:
        print("Query Error:", e)
        users = User.objects.select_related("category", "city").all()[:20]

    data = ""

    for u in users:

        try:
            category_name = u.category.name
        except:
            category_name = "Unknown"

        try:
            city_name = u.city.city
        except:
            city_name = "Unknown"

        data += f"Business: {u.business_name}, Category: {category_name}, City: {city_name}\n"

    return data
    

def detect_intent(user_msg):

    prompt = f"""
You are a Django ORM expert.

Convert user question into Django ORM filter in JSON.

Rules:
- If city mentioned → use city__city__icontains
- If category → category__name__icontains
- If name → business_name__icontains

Return ONLY JSON.

User Question: {user_msg}
"""

    intent = ask_llama(prompt)

    return intent.strip().lower()


def get_all_data():
    data_list = []

    users = User.objects.select_related("category","city").all()

    for u in users:
        
        try:
            category_name = u.category.name if u.category else "Unknown"
        except:
            category_name = "Unknown"

        try:
            city_name = u.city.city
        except:
            city_name = "Unknown"

        text = f"{u.business_name} is a {category_name} service in {city_name}"

        data_list.append(text)

    return data_list


DATA_LIST = get_all_data()
DATA_EMBEDDINGS = model.encode(DATA_LIST, convert_to_tensor=True)


def semantic_search(user_msg):

    global DATA_LIST, DATA_EMBEDDINGS
    msg = user_msg.lower()

    # 🔥 LOAD ONLY FIRST TIME
    if DATA_LIST is None:

        users = User.objects.select_related("category","city").all()
        DATA_LIST = []

        for u in users:
            try:
                category_name = u.category.name if u.category else "Unknown"
            except:
                category_name = "Unknown"

            try:
                city_name = u.city.city
            except:
                city_name = "Unknown"

            text = f"{u.business_name} is a {category_name} service in {city_name}"

            DATA_LIST.append(text)

        # embeddings only once
        DATA_EMBEDDINGS = model.encode(DATA_LIST, convert_to_tensor=True)

    # query embedding
    query_embedding = model.encode(msg, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, DATA_EMBEDDINGS)[0]
    sorted_results = scores.argsort(descending=True)
    results = ""

    for idx in sorted_results[:50]:
        results += DATA_LIST[int(idx)] + "\n"

    return results


def get_categories(page=1, limit=10):
    start = (page - 1) * limit
    end = start + limit

    total = Category.objects.count()
    categories = Category.objects.all()[start:end]

    data = []
    for c in categories:
        data.append(c.name)

    return {"total": total, "categories": data}

