from django.urls import path

from api import views

urlpatterns = [
    path("engines/", views.engines_list, name="engines-list"),
    path("game-review/", views.game_review, name="game-review"),
    path("position-explorer/", views.position_explorer, name="position-explorer"),
    path("bot-match/start/", views.bot_match_start, name="bot-match-start"),
    path("opening-lab/start/", views.opening_lab_start, name="opening-lab-start"),
]
