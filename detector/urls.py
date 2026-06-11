from django.urls import path

from .views import (
    upload_image,
    ai_camera,
    video_feed
)

urlpatterns = [

    path('', upload_image, name='upload'),

    path('ai-camera/', ai_camera, name='ai-camera'),

    path('video-feed/', video_feed, name='video-feed'),
]