from django.shortcuts import render
from django.http import JsonResponse

from chatapp.services.ai import ask_llama
from chatapp.services.utils import semantic_search, detect_intent, get_categories

from chatapp.models import Category
from chatapp.services.utils import get_dynamic_data

def chat_ui(request):
    return render(request, "chat.html")


def chatbot(request):

    user_msg = request.GET.get("msg")

    intent = detect_intent(user_msg)

    print("Intent:", intent)

    # CATEGORY
    if "categories" in intent:
        data = get_categories()
        return JsonResponse({
            "reply": f"Total Categories: {data['total']}\n\n" + "\n".join(data["categories"])
        })

    # PAGINATION
    elif "pagination" in intent:
        return JsonResponse({"reply": " Please type 'next' or 'prev'"})

    # AI SEARCH (MAIN)
    db_data = get_dynamic_data(user_msg)

    # fallback
    if not db_data.strip():
        db_data = semantic_search(user_msg)

    # still empty
    if not db_data.strip():
        return JsonResponse({
            "reply": "No matching businesses found . Try different keywords."
        })

    # LLM answer
    prompt = f"""
Answer ONLY from this data:
{db_data}

Question: {user_msg}
"""

    answer = ask_llama(prompt)

    return JsonResponse({"reply": answer})


def category_list(request):
    page = int(request.GET.get("page", 1))
    limit = 10

    start = (page - 1) * limit
    end = start + limit

    categories = Category.objects.all()[start:end]
    total = Category.objects.count()

    data = []

    for c in categories:
        data.append(c.name)

    return JsonResponse({"total": total, "categories": data})
