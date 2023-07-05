from django.urls import path
from .views import YouTubeDataAPIView

urlpatterns = [
    path('youtube/', YouTubeDataAPIView.as_view()),
]
