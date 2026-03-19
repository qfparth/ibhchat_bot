from sentence_transformers import SentenceTransformer, util
from chatapp.models import User


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




def get_all_data():
    data_list = []

    users = User.objects.select_related('category').all()

    for u in users:
        text = f"{u.business_name} is a {u.category.name} service in {u.city}"
        data_list.append(text)

    return data_list


model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_search(user_msg):
    data_list = get_all_data()

    data_embeddings = model.encode(data_list, convert_to_tensor=True)
    query_embedding = model.encode(user_msg, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, data_embeddings)[0]

    top_results = scores.topk(5)

    results = ""
    for idx in top_results.indices:
        results += data_list[int(idx)] + "\n"

    return results