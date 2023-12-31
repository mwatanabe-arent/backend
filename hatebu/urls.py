from django.urls import path
from .views import Topics, URL, TopicList, sandbox, character

urlpatterns = [
    path('hatebu/', Topics.as_view()),
    path('hatebu/list', TopicList.as_view()),
    path('hatebu/url', URL.as_view()),
    path('hatebu/sandbox', sandbox.as_view()),
    path('hatebu/character', character.as_view()),

]
