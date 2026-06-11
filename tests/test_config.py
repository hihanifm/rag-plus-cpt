from ragcpt.config import load_config


def test_config_loads_and_resolves_paths():
    cfg = load_config()
    assert cfg.base_model.startswith("Qwen/Qwen3-14B")
    assert cfg.teacher_model
    assert cfg.docs_dir.name == "docs"
    # dataset_mix present
    mix = cfg.get("dataset_mix")
    assert set(mix) == {"raw_text", "qa_plain", "qa_think", "qa_unknown"}


def test_get_default():
    cfg = load_config()
    assert cfg.get("nope", "missing", default="x") == "x"
