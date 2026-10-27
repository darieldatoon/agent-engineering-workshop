from kea.config import with_env_values


def test_fills_blank_line_copied_from_example():
    text = "LANGSMITH_API_KEY=\nLANGSMITH_TRACING=true\n"
    assert with_env_values(text, {"LANGSMITH_API_KEY": "k1"}) == (
        "LANGSMITH_API_KEY=k1\nLANGSMITH_TRACING=true\n"
    )


def test_uncomments_an_optional_line():
    text = "# LANGSMITH_GATEWAY_API_KEY=\nLANGSMITH_TRACING=true\n"
    assert with_env_values(text, {"LANGSMITH_GATEWAY_API_KEY": "g1"}) == (
        "LANGSMITH_GATEWAY_API_KEY=g1\nLANGSMITH_TRACING=true\n"
    )


def test_appends_a_missing_key_and_keeps_the_rest():
    text = "LANGSMITH_API_KEY=k1\nKEA_MODEL=openai:gpt-5.4-mini"
    assert with_env_values(text, {"LANGSMITH_GATEWAY_API_KEY": "g1"}) == (
        "LANGSMITH_API_KEY=k1\nKEA_MODEL=openai:gpt-5.4-mini\nLANGSMITH_GATEWAY_API_KEY=g1\n"
    )


def test_value_is_written_literally():
    # Keys are opaque strings; a backslash must not be read as a regex group reference.
    assert with_env_values("LANGSMITH_API_KEY=\n", {"LANGSMITH_API_KEY": r"a\1b"}) == (
        "LANGSMITH_API_KEY=a\\1b\n"
    )


def test_does_not_touch_a_key_whose_name_merely_ends_the_same():
    text = "LANGSMITH_GATEWAY_API_KEY=g1\n"
    assert with_env_values(text, {"LANGSMITH_API_KEY": "k1"}) == (
        "LANGSMITH_GATEWAY_API_KEY=g1\nLANGSMITH_API_KEY=k1\n"
    )
