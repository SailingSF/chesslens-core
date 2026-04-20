"""
Download an official Stockfish binary into ./engines/.

Defaults to version 16.1 (what chess.com uses). Auto-detects the current
platform. Binaries come from the official Stockfish GitHub releases.

Usage:
    python manage.py install_stockfish              # downloads 16.1
    python manage.py install_stockfish --version 17
    python manage.py install_stockfish --list       # show supported versions
"""

from __future__ import annotations

import os
import platform
import stat
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


# Release URLs for supported versions. Keep this small and explicit — users
# who need a different build can download it manually into ./engines/.
RELEASES = {
    "16.1": {
        "darwin-arm64": "https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-macos-m1-apple-silicon.tar",
        "darwin-x86_64": "https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-macos-x86-64-avx2.tar",
        "linux-x86_64": "https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-ubuntu-x86-64-avx2.tar",
        "windows-x86_64": "https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-windows-x86-64-avx2.zip",
    },
    "17": {
        "darwin-arm64": "https://github.com/official-stockfish/Stockfish/releases/download/sf_17/stockfish-macos-m1-apple-silicon.tar",
        "darwin-x86_64": "https://github.com/official-stockfish/Stockfish/releases/download/sf_17/stockfish-macos-x86-64-avx2.tar",
        "linux-x86_64": "https://github.com/official-stockfish/Stockfish/releases/download/sf_17/stockfish-ubuntu-x86-64-avx2.tar",
        "windows-x86_64": "https://github.com/official-stockfish/Stockfish/releases/download/sf_17/stockfish-windows-x86-64-avx2.zip",
    },
}


def _platform_key() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin":
        return "darwin-arm64" if machine in ("arm64", "aarch64") else "darwin-x86_64"
    if system == "linux":
        return "linux-x86_64"
    if system == "windows":
        return "windows-x86_64"
    raise CommandError(f"Unsupported platform: {system}/{machine}")


class Command(BaseCommand):
    help = "Download an official Stockfish binary into ./engines/."

    def add_arguments(self, parser):
        parser.add_argument("--version", default="16.1",
                            help="Stockfish version (default: 16.1)")
        parser.add_argument("--list", action="store_true",
                            help="List supported versions and exit")

    def handle(self, *args, version: str, list: bool, **options):
        if list:
            for v in sorted(RELEASES.keys()):
                self.stdout.write(f"  {v}")
            return

        if version not in RELEASES:
            raise CommandError(
                f"Version {version!r} not in the supported list. "
                f"Use --list to see options, or download manually into ./engines/."
            )

        plat = _platform_key()
        url = RELEASES[version].get(plat)
        if url is None:
            raise CommandError(f"No {version} build available for {plat}.")

        engines_dir = Path(settings.BASE_DIR) / "engines"
        engines_dir.mkdir(exist_ok=True)
        target = engines_dir / f"stockfish-{version}"
        if platform.system().lower() == "windows":
            target = target.with_suffix(".exe")

        if target.exists():
            self.stdout.write(self.style.WARNING(
                f"{target} already exists. Delete it first if you want to re-download."
            ))
            return

        self.stdout.write(f"Downloading Stockfish {version} for {plat}…")
        self.stdout.write(f"  {url}")

        with tempfile.TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / os.path.basename(url)
            urllib.request.urlretrieve(url, archive)

            extract_dir = Path(tmpdir) / "extract"
            extract_dir.mkdir()
            if archive.suffix == ".zip":
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(extract_dir)
            else:
                with tarfile.open(archive) as tf:
                    tf.extractall(extract_dir)

            binary = _find_binary(extract_dir)
            if binary is None:
                raise CommandError(f"Could not locate Stockfish binary inside {archive.name}.")

            target.write_bytes(binary.read_bytes())

        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.stdout.write(self.style.SUCCESS(f"Installed: {target}"))
        self.stdout.write("Run `python manage.py list_engines` to verify discovery.")


def _find_binary(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if name.startswith("stockfish") and not name.endswith((".txt", ".md", ".pdf", ".html")):
            return p
    return None
