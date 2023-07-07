from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from common.utils import message_response,make_questions

@require_GET
def hello(request):
    name = request.GET.get('name', '')
    return JsonResponse({'message': f'Hello1, {name}!'})

@require_GET
def talkQuestion(request):
    message = request.GET.get('message', '')
    print(message)
    response = message_response(message)
    questions = make_questions(response)
    retdata = {
        "message":response,
        "question_json":questions
    }

    return JsonResponse(retdata)

