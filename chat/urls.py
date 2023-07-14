from django.urls import path
from . import views

urlpatterns = [
    path('chat/hello', views.hello, name='hello'),
    path('chat/', views.talkQuestion, name='talkQuestion'),
    path('chat/memory', views.Memory.as_view()),
]
