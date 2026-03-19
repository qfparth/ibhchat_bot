from django.urls import path
from .views import chatbot, chat_ui, category_list



urlpatterns = [
    path('', chat_ui),
    path('chatbot/', chatbot),
    path('categories/', category_list),
]