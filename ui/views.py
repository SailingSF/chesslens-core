from django.shortcuts import render


def index(request):
    return render(request, "ui/index.html")


def game_review(request):
    return render(request, "ui/game_review.html")


def position_explorer(request):
    return render(request, "ui/position_explorer.html")
