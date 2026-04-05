from pathlib import Path

from tests.optimize_classification import (
    CacheSpec,
    _discover_cache_specs,
    _missing_games_for_cache_spec,
    _parse_cache_spec,
)


def test_parse_versioned_node_cache_spec():
    spec = _parse_cache_spec(Path("game_166644797302_sf16.1_n2000000_pv3.json"))

    assert spec == CacheSpec(engine_tag="sf16.1", nodes=2_000_000, multipv=3)


def test_parse_legacy_depth_cache_spec():
    spec = _parse_cache_spec(Path("game_01_stockfish_d18_pv3.json"))

    assert spec == CacheSpec(engine_tag="stockfish", depth=18, multipv=3)


def test_discover_cache_specs_ignores_non_cache_files():
    specs = _discover_cache_specs([
        Path("game_01_sf16.1_n2000000_pv3.json"),
        Path("game_02_stockfish_d18_pv3.json"),
        Path("game_elos.json"),
    ])

    assert specs == [
        CacheSpec(engine_tag="sf16.1", nodes=2_000_000, multipv=3),
        CacheSpec(engine_tag="stockfish", depth=18, multipv=3),
    ]


def test_missing_games_for_cache_spec_handles_versioned_and_legacy_names(tmp_path):
    game_files = []
    for stem in ("game_01", "game_02", "game_03"):
        csv_path = tmp_path / f"{stem}.csv"
        csv_path.write_text("placeholder\n")
        game_files.append(csv_path)

    (tmp_path / "game_01_sf16.1_n2000000_pv3.json").write_text("[]\n")
    (tmp_path / "game_02_stockfish_d18_pv3.json").write_text("[]\n")

    versioned_missing = _missing_games_for_cache_spec(
        game_files,
        CacheSpec(engine_tag="sf16.1", nodes=2_000_000, multipv=3),
    )
    legacy_missing = _missing_games_for_cache_spec(
        game_files,
        CacheSpec(engine_tag="stockfish", depth=18, multipv=3),
    )

    assert [path.name for path in versioned_missing] == ["game_02.csv", "game_03.csv"]
    assert [path.name for path in legacy_missing] == ["game_01.csv", "game_03.csv"]
