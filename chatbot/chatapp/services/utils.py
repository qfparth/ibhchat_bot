from sentence_transformers import SentenceTransformer, util
from chatapp.models import User
from chatapp.models import Category
from chatapp.services.ai import ask_llama


model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_query(user_msg):

    prompt = f"""
You are a Django ORM expert.

Convert user question into Django ORM filter.

STRICT RULES:
- Only return filter
- No sentence
- No explanation
- No "Here is..."
- No backticks `
- Output must be ONE LINE

Example:
city__icontains="ahmedabad",category__name__icontains="apparel"

User Question: {user_msg}
"""

    query = ask_llama(prompt)

    return query.strip() if query else ""



def get_dynamic_data(user_msg):

    allowed_fields = [
    "city__icontains",
    "city__city__icontains",
    "category__name__icontains",
    "business_name__icontains"
]



    query = generate_query(user_msg)
    if not query:
        print("AI returned None")
        return ""
    
    query = query.replace("```", "")
    query = query.replace("python", "")
    query = query.replace("Here is the Django ORM filter:", "")
    query = query.replace("Here is", "")
    query = query.replace("filter:", "")
    query = query.replace("`", "")
    query = query.strip()

    try:
        filter_dict = {}
        parts = query.split(",")

        for p in parts:
            if "=" in p:
                key, value = p.split("=", 1)

                key = key.strip()
                value = value.strip().replace('"', '')
                if key in allowed_fields:
                    filter_dict[key] = value

        # AUTO FIX FOR CITY
        if "city__icontains" in filter_dict:
            filter_dict["city__city__icontains"] = filter_dict["city__icontains"]
            del filter_dict["city__icontains"]

        users = User.objects.select_related("category").filter(
            **filter_dict, category__isnull=False
        )[:20]

    except Exception as e:
        print("Query Error:", e)
        users = User.objects.all()[:20]

    data = ""

    for u in users:
        category_name = u.category.name if u.category else "Unknown"

        data += f"Business: {u.business_name}, Category: {category_name}, City: {u.city.city}\n"

    return data
    

def detect_intent(user_msg):

    prompt = f"""
Classify this user message into ONE of these:

- categories
- pagination
- search
- normal

Message: {user_msg}

Only return the intent name.
"""

    intent = ask_llama(prompt)

    return intent.strip().lower()


def get_all_data():
    data_list = []

    users = User.objects.select_related("category").all()

    for u in users:
        try:
            category_name = u.category.name if u.category else "Unknown"
            text = f"{u.business_name} is a {category_name} service in {u.city.city}"

        except:
            text = f"{u.business_name} is a service in {u.city.city}"

        data_list.append(text)

    return data_list


DATA_LIST = get_all_data()
DATA_EMBEDDINGS = model.encode(DATA_LIST, convert_to_tensor=True)


def semantic_search(user_msg):

    global DATA_LIST, DATA_EMBEDDINGS
    msg = user_msg.lower()

    # 🔥 LOAD ONLY FIRST TIME
    if DATA_LIST is None:

        users = User.objects.select_related("category").all()
        DATA_LIST = []

        for u in users:
            try:
                category_name = u.category.name if u.category else "Unknown"
                text = f"{u.business_name} is a {category_name} service in {u.city.city}"

            except:
                text = f"{u.business_name} is a service in {u.city}"

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

