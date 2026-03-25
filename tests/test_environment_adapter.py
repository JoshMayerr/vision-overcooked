from vision_overcooked.adapters.environment import EnvironmentAdapter


def test_environment_adapter_lists_tasks():
    adapter = EnvironmentAdapter()
    tasks = adapter.available_tasks()
    assert any(task["order"] == "boiled_egg" for task in tasks)


def test_environment_adapter_renders_frame():
    adapter = EnvironmentAdapter(horizon=4)
    snapshot = adapter.reset("boiled_egg")
    assert snapshot.frame.ndim == 3
    assert snapshot.frame.shape[2] == 3
    assert snapshot.state_string
