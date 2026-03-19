from django.shortcuts import render
from django.http import JsonResponse
from chatapp.services.utils import get_data
from chatapp.services.ai import ask_llama

from chatapp.services.utils import semantic_search
from django.http import JsonResponse

def chat_ui(request):
    return render(request, "chat.html")

def chatbot(request):
    user_msg = request.GET.get("msg")

    db_data = get_data(user_msg)

    prompt = f"""
You are a strict database assistant.

Rules:
1. Answer ONLY from given data.
2. Do NOT use your own knowledge.
3. Do NOT explain anything extra.
4. If data not found → say "No data found in database".

DATA:
{db_data}

Question: {user_msg}
"""

    answer = ask_llama(prompt)

    return JsonResponse({"reply": answer})

def chatbot(request):
    user_msg = request.GET.get("msg")

    # AI search
    db_data = semantic_search(user_msg)

    # strict prompt
    prompt = f"""
You are a strict assistant.

Answer ONLY from this data:
{db_data}

Question: {user_msg}
"""

    answer = ask_llama(prompt)

    return JsonResponse({"reply": answer})