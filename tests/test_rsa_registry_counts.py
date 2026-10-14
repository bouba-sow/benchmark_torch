"""Historical checkpoints must not inflate main-model RSA coverage."""
import pandas as pd

from neuroai.analysis.benchmark_common import filter_registry_models


def test_registry_filter_preserves_exact_main_and_rejects_checkpoint_aliases():
    frame = pd.DataFrame([
        ('pythia', 'EleutherAI/pythia-14m-deduped', 'main', 0.1),
        ('pythia_checkpoints', 'EleutherAI/pythia-14m-deduped', 'main', 0.2),
        ('pythia', 'EleutherAI/pythia-14m-deduped', 'step143000', 0.3),
        ('pythia', 'unknown/model', 'main', 0.4),
    ], columns=['family', 'model_id', 'revision', 'rsa_pearson'])
    result = filter_registry_models(frame)
    assert result.index.tolist() == [0]
    assert result.rsa_pearson.tolist() == [0.1]
    assert len(frame) == 4


def test_behavior_rsa_retries_invalid_result_and_rejects_nonfinite_input(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    import numpy as np
    import pytest

    path = Path(__file__).resolve().parents[1] / 'scripts/analysis/surprisal_rsa.py'
    spec = importlib.util.spec_from_file_location('rsa_count_test_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / 'spr/gpt2/openai-community_gpt2/rsa_main.csv'
    out.parent.mkdir(parents=True)
    pd.DataFrame({'rsa_pearson': [np.nan]}).to_csv(out, index=False)
    values = pd.DataFrame({'story_id': [1, 1, 1], 'zone': [1, 2, 3],
                           'surprisal': [1., np.nan, 3.]})
    monkeypatch.setattr(module, '_load_behavior_surprisal', lambda *args: values)
    responses = {'SPR': values[['story_id', 'zone']].assign(rt=[100., 200., 150.])}
    with pytest.raises(ValueError, match='nonfinite aligned'):
        module._score_behavior_model(('gpt2', 'openai-community/gpt2', 'main'),
                                     'naturalstories', responses, tmp_path, ('SPR',))
    assert not (tmp_path / 'rdm_cache').exists()


def test_rsa_report_keeps_extreme_final_layer_likelihood(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / 'scripts/analysis/build_rsa_report.py'
    spec = importlib.util.spec_from_file_location('rsa_report_test_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    score_path = (tmp_path / 'outputs_jeanzay/onestop/surprisal_sweep/sp/gpt2'
                  / 'openai-community_gpt2/onestop_lme_checkpoint_scores.csv')
    score_path.parent.mkdir(parents=True)
    pd.DataFrame({'revision': ['main'] * 5, 'measure': ['SP'] * 5,
                  'layer': [0, 1, 2, 3, 4], 'delta_loglik_test': [1., 2., 3., 4., 1000.],
                  'n_obs_test': [100] * 5}).to_csv(score_path, index=False)
    frame = pd.DataFrame([('gpt2', 'openai-community/gpt2', 'main')], columns=module.KEYS)
    refs = module.likelihood_reference('onestop', 'SP', [frame])
    assert refs[('gpt2', 'openai-community/gpt2', 'main')][0] == 10.


def test_naturalstories_complete_trial_cache_requires_matching_word_order(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    import numpy as np
    import pytest
    from neuroai.datasets.natural_stories import NaturalStoriesDataset

    path = Path(__file__).resolve().parents[1] / 'scripts/analysis/surprisal_rsa.py'
    spec = importlib.util.spec_from_file_location('rsa_fallback_test_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.delenv('NS_FEATURE_CACHE_DIR', raising=False)
    monkeypatch.setenv('SCRATCH', str(tmp_path / 'scratch'))
    cache = tmp_path / 'scratch/.cache/naturalstories_features'
    cache.mkdir(parents=True)
    values = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    from neuroai.features.logit_lens import _trial_surprisal_cache_path
    np.savez(cache / _trial_surprisal_cache_path('.', 'openai-community/gpt2', 'main', 'natural_stories', False, 0).name,
             features=values)
    words = pd.DataFrame({'story_id': [1, 1], 'zone': [1, 2]})
    monkeypatch.setattr(NaturalStoriesDataset, 'load_words', lambda self: words.copy())
    result = module._naturalstories_trial_fallback(words, 'openai-community/gpt2', 'main')
    np.testing.assert_array_equal(result, values)
    with pytest.raises(ValueError, match='word ordering mismatch'):
        module._naturalstories_trial_fallback(words.iloc[::-1], 'openai-community/gpt2', 'main')
