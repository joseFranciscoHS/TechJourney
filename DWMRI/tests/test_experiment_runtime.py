"""Unit tests for utils.experiment_runtime override parsing."""

from utils.experiment_runtime import apply_overrides, parse_override_value


def test_parse_list_of_ints():
    assert parse_override_value("[2,4,10]") == [2, 4, 10]


def test_parse_list_with_spaces():
    assert parse_override_value(" [1, 2, 4] ") == [1, 2, 4]


def test_parse_dict_literal():
    assert parse_override_value("{'a': 1}") == {"a": 1}


def test_parse_scalars():
    assert parse_override_value("24") == 24
    assert parse_override_value("2.66") == 2.66
    assert parse_override_value("true") is True
    assert parse_override_value("false") is False
    assert parse_override_value("null") is None
    assert parse_override_value("none") is None


def test_parse_plain_string_unchanged():
    assert parse_override_value("res_cnn_2d") == "res_cnn_2d"


def test_parse_invalid_bracket_falls_back_to_string():
    # Not a valid literal; must not raise, must return the raw string.
    assert parse_override_value("[not, valid") == "[not, valid"


def test_apply_overrides_writes_nested_list():
    settings = {"dbrain": {"model": {"num_blocks": [1, 2, 2]}}}
    apply_overrides(settings, ["dbrain.model.num_blocks=[2,4,10]"])
    assert settings["dbrain"]["model"]["num_blocks"] == [2, 4, 10]
