import numpy as np

from neuroai.analysis.pereira_rsa import prepare_neural_rdms, score_last_layer


def test_identical_geometry_has_high_pearson_rsa():
    rng = np.random.default_rng(0)
    states = rng.normal(size=(12, 8))
    item_ids = np.array([f"384sentences.{i}" for i in range(len(states))])
    neural = {"001": (states + rng.normal(scale=0.01, size=states.shape), item_ids, np.array(["ROI"] * 8))}
    rdms = prepare_neural_rdms(item_ids, neural)
    result = score_last_layer(states, rdms, model_id="test/model", revision="main")
    assert len(result) == 1
    assert result.rsa_pearson.iloc[0] > 0.99
