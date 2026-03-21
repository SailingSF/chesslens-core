"""Django Channels WebSocket URL routing."""

from django.urls import path

from ws.consumers import BotMatchConsumer, OpeningLabConsumer

websocket_urlpatterns = [
    path("ws/bot-match/<str:session_id>/", BotMatchConsumer.as_asgi()),
    path("ws/opening-lab/<str:session_id>/", OpeningLabConsumer.as_asgi()),
]
