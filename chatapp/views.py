from django.shortcuts import render
from django.http import JsonResponse

from chatapp.services.ai import ask_llama
from chatapp.services.utils import semantic_search, detect_intent, get_categories
from chatapp.services.utils import get_dynamic_data
from chatapp.services.utils import chat

from chatapp.models import Category


def chat_ui(request):
    return render(request, "chat.html")


def chatbot(request):
    user_msg = request.GET.get("msg", "").strip()
    if not user_msg:
        return JsonResponse({"reply": "I didn't catch that. Please type your question."})
    reply = chat(user_msg)
    return JsonResponse({"reply": reply})


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