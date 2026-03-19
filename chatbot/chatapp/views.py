from django.shortcuts import render
from django.http import JsonResponse
from chatapp.services.utils import get_data
from chatapp.services.ai import ask_llama

from chatapp.services.utils import semantic_search

from chatapp.services.utils import get_categories

from chatapp.models import Category

def chat_ui(request):
    return render(request, "chat.html")

# ✅ keep this
def chatbot(request):
    user_msg = request.GET.get("msg")

    db_data = semantic_search(user_msg)

    prompt = f"""
You are a strict assistant.

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

    return JsonResponse({
        "total": total,
        "categories": data
    })
