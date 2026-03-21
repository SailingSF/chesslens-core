"""
Django models for the game app.

Stores completed game analysis results for history and retrieval.
"""

from django.db import models


class SavedGame(models.Model):
    """A completed game with its PGN and analysis metadata."""

    pgn = models.TextField()
    white_player = models.CharField(max_length=128, blank=True)
    black_player = models.CharField(max_length=128, blank=True)
    result = models.CharField(max_length=8, blank=True)   # "1-0", "0-1", "1/2-1/2"
    created_at = models.DateTimeField(auto_now_add=True)
    analyzed_at = models.DateTimeField(null=True, blank=True)
    skill_level = models.CharField(max_length=16, default="intermediate")

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.white_player} vs {self.black_player} ({self.result})"


class MoveNote(models.Model):
    """LLM-generated explanation for a single move in a saved game."""

    game = models.ForeignKey(SavedGame, on_delete=models.CASCADE, related_name="notes")
    move_number = models.PositiveIntegerField()
    color = models.CharField(max_length=8)   # "white" or "black"
    move_san = models.CharField(max_length=16)
    fen = models.TextField()
    cp_loss = models.IntegerField(null=True, blank=True)
    cp_loss_label = models.CharField(max_length=16, blank=True)
    priority_tier = models.CharField(max_length=16, blank=True)
    explanation = models.TextField(blank=True)

    class Meta:
        ordering = ["move_number", "color"]
        unique_together = [("game", "move_number", "color")]

    def __str__(self):
        return f"Move {self.move_number} {self.color}: {self.move_san}"
