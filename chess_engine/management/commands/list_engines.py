"""Print discovered Stockfish engines."""

from django.core.management.base import BaseCommand

from chess_engine.discovery import choose_default, discover_engines, reset_cache


class Command(BaseCommand):
    help = "List Stockfish binaries discovered on this machine."

    def handle(self, *args, **options):
        reset_cache()
        engines = discover_engines()
        if not engines:
            self.stdout.write(self.style.WARNING(
                "No Stockfish engines found.\n"
                "Drop a binary into ./engines/ or run "
                "`python manage.py install_stockfish` to download one."
            ))
            return

        default = choose_default(engines)
        for e in engines:
            marker = " (default)" if default and e.id == default.id else ""
            self.stdout.write(f"  {e.id}{marker}")
            self.stdout.write(f"    name:    {e.name}")
            self.stdout.write(f"    path:    {e.path}")
