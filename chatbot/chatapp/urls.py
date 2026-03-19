from django.urls import path
from .views import chatbot, chat_ui



urlpatterns = [
    path('', chat_ui),
    path('chatbot/', chatbot),
]