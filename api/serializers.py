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


class BotMatchStartSerializer(serializers.Serializer):
    elo = serializers.IntegerField(min_value=400, max_value=3000, default=1200)
    player_color = serializers.ChoiceField(choices=["white", "black"], default="white")
    coaching_mode = serializers.ChoiceField(
        choices=["silent", "nudge", "verbose"], default="nudge"
    )
    skill_level = serializers.ChoiceField(choices=SKILL_LEVELS, default="intermediate")
    opening = serializers.CharField(required=False, allow_blank=True)


class OpeningLabStartSerializer(serializers.Serializer):
    eco_code = serializers.CharField(max_length=4)
    player_color = serializers.ChoiceField(choices=["white", "black"], default="white")
    skill_level = serializers.ChoiceField(choices=SKILL_LEVELS, default="intermediate")
