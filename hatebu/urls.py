from django.urls import path
from .views import Topics
from .views import URL

urlpatterns = [
    path('hatebu/', Topics.as_view()),
    path('hatebu/url', URL.as_view()),

]
