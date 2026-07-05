from rest_framework import serializers

SKILL_LEVELS = ["beginner", "intermediate", "advanced"]
LLM_PROVIDERS = ["anthropic", "openai"]
REASONING_EFFORTS = ["low", "medium", "high"]


class GameReviewSerializer(serializers.Serializer):
    pgn = serializers.CharField()
    skill_level = serializers.ChoiceField(choices=SKILL_LEVELS, default="intermediate")
    player_color = serializers.ChoiceField(choices=["white", "black"], default="white")
    engine_id = serializers.CharField(required=False, allow_blank=True)
    llm_provider = serializers.ChoiceField(choices=LLM_PROVIDERS, required=False)
    llm_model = serializers.CharField(required=False, allow_blank=True)
    llm_reasoning_effort = serializers.ChoiceField(
        choices=REASONING_EFFORTS, required=False
    )


class ChessComImportSerializer(serializers.Serializer):
    username = serializers.RegexField(
        r"^[A-Za-z0-9_-]{1,50}$",
        error_messages={"invalid": "Invalid chess.com username."},
    )


class PositionExplorerSerializer(serializers.Serializer):
    fen = serializers.CharField()
    skill_level = serializers.ChoiceField(choices=SKILL_LEVELS, default="intermediate")
    question = serializers.CharField(required=False, allow_blank=True)
    engine_id = serializers.CharField(required=False, allow_blank=True)
    llm_provider = serializers.ChoiceField(choices=LLM_PROVIDERS, required=False)
    llm_model = serializers.CharField(required=False, allow_blank=True)
    llm_reasoning_effort = serializers.ChoiceField(
        choices=REASONING_EFFORTS, required=False
    )


class ChatMessageSerializer(serializers.Serializer):
    role = serializers.ChoiceField(choices=["user", "assistant"])
    content = serializers.CharField()


class PositionChatSerializer(serializers.Serializer):
    fen = serializers.CharField()
    skill_level = serializers.ChoiceField(choices=SKILL_LEVELS, default="intermediate")
    messages = serializers.ListField(child=ChatMessageSerializer(), allow_empty=False)
    engine_id = serializers.CharField(required=False, allow_blank=True)
    llm_provider = serializers.ChoiceField(choices=LLM_PROVIDERS, required=False)
    llm_model = serializers.CharField(required=False, allow_blank=True)
    llm_reasoning_effort = serializers.ChoiceField(
        choices=REASONING_EFFORTS, required=False
    )

    def validate_messages(self, value):
        if value[-1]["role"] != "user":
            raise serializers.ValidationError("The last message must be from the user.")
        return value
