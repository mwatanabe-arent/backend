# Generated by Django 4.2.1 on 2023-07-13 07:18

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='bodytext',
            name='user',
        ),
    ]
