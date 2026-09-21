import json

from export.hf_runtime_assets import copy_hf_runtime_assets


def test_copies_processor_assets_from_recorded_base(tmp_path):
    base = tmp_path / "base"
    clean = tmp_path / "clean"
    exported = tmp_path / "exported"
    base.mkdir()
    clean.mkdir()
    (base / "preprocessor_config.json").write_text("image")
    (base / "video_preprocessor_config.json").write_text("video")
    (base / "config.json").write_text("do-not-copy")
    (clean / "clean_hf_dump_summary.json").write_text(
        json.dumps({"base_model_path": str(base)})
    )

    copied = copy_hf_runtime_assets(clean, exported)

    assert copied == ["preprocessor_config.json", "video_preprocessor_config.json"]
    assert (exported / "preprocessor_config.json").read_text() == "image"
    assert (exported / "video_preprocessor_config.json").read_text() == "video"
    assert not (exported / "config.json").exists()


def test_does_not_overwrite_exported_asset(tmp_path):
    source = tmp_path / "source"
    exported = tmp_path / "exported"
    source.mkdir()
    exported.mkdir()
    (source / "preprocessor_config.json").write_text("source")
    (exported / "preprocessor_config.json").write_text("exported")

    assert copy_hf_runtime_assets(source, exported) == []
    assert (exported / "preprocessor_config.json").read_text() == "exported"
