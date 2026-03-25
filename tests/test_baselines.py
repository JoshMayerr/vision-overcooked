from vision_overcooked.baselines import baseline_summary, load_literature_baselines


def test_baseline_seed_file_loads():
    records = load_literature_baselines()
    assert len(records) >= 2
    assert all(record.source_id == "sun_etal_2025_collab_overcooked" for record in records)


def test_baseline_summary_contains_models():
    summary = baseline_summary()
    assert summary["record_count"] >= 2
    assert "GPT-4o" in summary["models"]
