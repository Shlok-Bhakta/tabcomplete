from pathlib import Path

import yaml


def test_registered_r2_budget_and_shared_treatment_settings():
    root = Path(__file__).parents[1]
    config = yaml.safe_load((root / "configs/research/model_data_r2.yaml").read_text())
    train = config["training"]
    assert train["successful_updates"] * train["tokens_per_update"] == 5_013_504
    assert train["parent"] == "q35-p12"
    assert 2 * 5_013_504 + (6 + 16) * 32768 < config["limits"]["additional_training_input_tokens"]
    assert config["evaluation"]["causal"]["max_new_tokens"] == 96
    assert config["limits"]["simultaneous_gpu_allocations"] == 1
    assert config["selection"]["deployed_model_replacement"] == "forbidden"
    assert config["long_context"]["gpu_wall_seconds"] == 3600
