from django.shortcuts import render


def index(request):
    return render(request, "ui/index.html", {"active_page": "home"})


def game_review(request):
    return render(request, "ui/game_review.html", {"active_page": "review"})


def position_explorer(request):
    return render(request, "ui/position_explorer.html", {"active_page": "explorer"})
