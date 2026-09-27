"""--config TOML: same options as the CLI, CLI flags win, loud failures on bad files."""

from __future__ import annotations

import pytest

from cali_address.io import ConfigError
from cali_address.io.config import load_config, merge_options


def _toml(tmp_path, text, name="cfg.toml"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_load_flat_config_with_dashes_or_underscores(tmp_path):
    cfg = load_config(_toml(tmp_path, 'address-col = "direccion"\nchunk_size = 500\nno_gazetteer = true\n'))
    assert cfg == {"address_col": "direccion", "chunk_size": 500, "no_gazetteer": True}


def test_address_parts_accept_array_or_string(tmp_path):
    a = load_config(_toml(tmp_path, 'address_parts = ["via", "numero"]\n'))
    b = load_config(_toml(tmp_path, 'address_parts = "via,numero"\n', "b.toml"))
    assert a["address_parts"] == ["via", "numero"] and b["address_parts"] == ["via", "numero"]


def test_nested_tables_are_flattened(tmp_path):
    cfg = load_config(_toml(tmp_path, '[input]\nsheet = "hoja"\nheader_row = 2\n[tunables]\nmin_struct = 0.7\n'))
    assert cfg == {"sheet": "hoja", "header_row": 2, "min_struct": 0.7}


def test_malformed_toml_is_config_error(tmp_path):
    with pytest.raises(ConfigError, match="TOML"):
        load_config(_toml(tmp_path, 'address_col = = "x"\n'))


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError):
        load_config(str(tmp_path / "nope.toml"))


def test_unknown_key_lists_valid_ones(tmp_path):
    with pytest.raises(ConfigError, match="chunk_size"):
        load_config(_toml(tmp_path, 'chunk_sise = 10\n'))


@pytest.mark.parametrize("text", ['chunk_size = "many"\n', 'min_struct = "high"\n', 'no_gazetteer = "yes"\n',
                                  'header_row = 1.5\n', 'address_col = 5\n', 'chunk_size = 0\n'])
def test_wrong_types_or_values_are_config_errors(tmp_path, text):
    with pytest.raises(ConfigError):
        load_config(_toml(tmp_path, text))


def test_precedence_cli_over_file_over_defaults():
    defaults = {"chunk_size": 20_000, "sheet": None, "min_struct": 0.6}
    file_cfg = {"chunk_size": 500, "sheet": "hoja"}
    cli = {"chunk_size": 50, "sheet": None, "min_struct": None}
    merged = merge_options(defaults, file_cfg, cli)
    assert merged == {"chunk_size": 50, "sheet": "hoja", "min_struct": 0.6}


def test_cli_false_boolean_overrides_file_true():
    merged = merge_options({"gazetteer": True}, {"gazetteer": False}, {"gazetteer": True})
    assert merged["gazetteer"] is True
    merged = merge_options({"gazetteer": True}, {"gazetteer": False}, {"gazetteer": None})
    assert merged["gazetteer"] is False
