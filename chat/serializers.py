

from rest_framework import serializers
from itsdangerous import Serializer
from .models import BodyText, User


class ChatBuffersSerializer(Serializer.ModelSerializer):
    group_name = serializers.SerializerMethodField()

    class Meta:
        model = BodyText
        fields = ('id', 'username', 'body')
