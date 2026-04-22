from mechinterp.experiments.run_patching import _aggregate_patch_rows, _selected_positions


def test_selected_positions_all_and_final_modes() -> None:
    assert _selected_positions(4, "all") == [0, 1, 2, 3]
    assert _selected_positions(4, "final") == [3]


def test_aggregate_patch_rows_builds_grid_and_top_entries() -> None:
    rows = [
        {"layer": 0, "position": 0, "token": "<|endoftext|>", "normalized_effect": 0.1},
        {"layer": 0, "position": 1, "token": "When", "normalized_effect": 0.2},
        {"layer": 1, "position": 1, "token": "When", "normalized_effect": 0.8},
        {"layer": 1, "position": 1, "token": "When", "normalized_effect": 0.6},
    ]

    aggregate = _aggregate_patch_rows(rows)

    assert aggregate["layer_labels"] == [0, 1]
    assert aggregate["position_labels"] == ["0:<|endoftext|>", "1:When"]
    assert aggregate["mean_normalized_effect"] == [[0.1, 0.2], [None, 0.7]]
    assert aggregate["top_entries"][0] == {
        "layer": 1,
        "position": 1,
        "mean_normalized_effect": 0.7,
    }
