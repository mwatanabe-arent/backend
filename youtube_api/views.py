from django.shortcuts import render

# Create your views here.
from googleapiclient.discovery import build
from rest_framework.views import APIView
from rest_framework.response import Response

class YouTubeDataAPIView(APIView):
    def get(self, request):
        return Response("Hello, world!api2")
        api_key = "YOUR_API_KEY"  # 自分のAPIキーに書き換える
        youtube = build('youtube', 'v3', developerKey=api_key)

        request = youtube.search().list(
            part="snippet",
            maxResults=25,
            q=request.query_params.get('q')  # クエリパラメーターから検索文字列を取得
        )
        response = request.execute()

        return Response(response)
