from django.urls import path

from api import views

urlpatterns = [
    path("engines/", views.engines_list, name="engines-list"),
    path("game-review/", views.game_review, name="game-review"),
    path("position-explorer/", views.position_explorer, name="position-explorer"),
    path("position-chat/", views.position_chat, name="position-chat"),
    path("chesscom-import/", views.chesscom_import, name="chesscom-import"),
    path("imported-games/", views.imported_games_list, name="imported-games-list"),
    path(
        "imported-games/<str:username>/<str:game_id>/",
        views.imported_game_pgn,
        name="imported-game-pgn",
    ),
]
