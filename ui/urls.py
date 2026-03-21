from django.urls import path

from ui import views

app_name = "ui"

urlpatterns = [
    path("", views.index, name="index"),
    path("game-review/", views.game_review, name="game-review"),
    path("position/", views.position_explorer, name="position-explorer"),
]
