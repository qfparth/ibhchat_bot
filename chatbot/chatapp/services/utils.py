from sentence_transformers import SentenceTransformer, util
from chatapp.models import User
from chatapp.models import Category
from chatapp.services.ai import ask_llama

model = SentenceTransformer("all-MiniLM-L6-v2")


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
            text = f"{u.business_name} is a {u.category.name} service in {u.city}"
        except:
            text = f"{u.business_name} is a service in {u.city}"

        data_list.append(text)

    return data_list


DATA_LIST = get_all_data()
DATA_EMBEDDINGS = model.encode(DATA_LIST, convert_to_tensor=True)


# --------- SentanceTransfor ------------
model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_search(user_msg):

    global DATA_LIST, DATA_EMBEDDINGS

    msg = user_msg.lower()

    # 🔥 LOAD ONLY FIRST TIME
    if DATA_LIST is None:

        users = User.objects.select_related("category").all()

        DATA_LIST = []

        for u in users:
            try:
                text = f"{u.business_name} is a {u.category.name} service in {u.city}"
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


def get_company_data():
    data_list = []

    companies = User.objects.select_related("category").all()

    for c in companies:
        text = f"{c.business_name} is a {c.category.name} company in {c.city}"
        data_list.append(text)

    return data_list


def get_categories(page=1, limit=10):
    start = (page - 1) * limit
    end = start + limit

    total = Category.objects.count()
    categories = Category.objects.all()[start:end]

    data = []
    for c in categories:
        data.append(c.name)

    return {"total": total, "categories": data}


def get_data(user_msg):
    data = ""
    msg = user_msg.lower()

    # lowercase માં compare કરવું
    if "ahmedabad" in msg:
        users = User.objects.filter(city__icontains="ahmedabad")

    elif "apparel" in msg:
        users = User.objects.filter(category__name__icontains="apparel")

    else:
        users = User.objects.all()[:20]

    # 👉SAFE LOOP
    for u in users[:20]:
        try:
            category_name = u.category.name
        except:
            category_name = "Unknown"

        data += f"Business: {u.business_name}, Category: {u.category.name}, City: {u.city}\n"

    return data