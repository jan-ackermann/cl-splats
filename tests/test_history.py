"""Tests for compact history recording and exact scene-state recovery.

clsplats.history imports gsplat only lazily (inside export_raw_ply), so no
gsplat mocking is needed here.
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clsplats.history import (  # noqa: E402
    PARAM_KEYS,
    HistoryRecorder,
    recover_state,
    revert_timestep,
)


def _make_params(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "means": torch.randn(n, 3, generator=g),
        "scales": torch.randn(n, 3, generator=g),
        "quats": torch.randn(n, 4, generator=g),
        "opacities": torch.randn(n, generator=g),
        "sh0": torch.randn(n, 1, 3, generator=g),
        "shN": torch.randn(n, 0, 3, generator=g),
    }


def _simulate_timestep(params, active, recorder, timestep, seed):
    """Mimic one CL timestep: optimise active rows, split one, densify children.

    Inactive rows are untouched and keep their relative order — the invariant
    the recovery relies on.
    """
    recorder.begin_timestep(timestep, active, params)

    g = torch.Generator().manual_seed(seed)
    new_params = {}
    n = active.shape[0]
    # "Optimise": perturb active rows only.
    for key in PARAM_KEYS:
        p = params[key].clone()
        noise = torch.randn(p[active].shape, generator=g)
        p[active] = p[active] + noise
        new_params[key] = p
    new_active = active.clone()

    # "Split" the first active row: parent removed mid-array, 2 children appended.
    split_idx = int(torch.nonzero(active)[0])
    keep = torch.ones(n, dtype=torch.bool)
    keep[split_idx] = False
    for key in PARAM_KEYS:
        parent = new_params[key][split_idx : split_idx + 1]
        children = torch.cat([parent, parent], dim=0) + 0.1
        new_params[key] = torch.cat([new_params[key][keep], children], dim=0)
    new_active = torch.cat([new_active[keep], torch.ones(2, dtype=torch.bool)])

    recorder.end_timestep(new_active)
    return new_params, new_active


def test_single_timestep_recovery_is_exact():
    params0 = _make_params(6)
    active = torch.tensor([False, True, False, True, False, False])
    recorder = HistoryRecorder()

    params1, _ = _simulate_timestep(params0, active, recorder, timestep=1, seed=1)

    recovered = revert_timestep(params1, recorder.records[0])
    for key in PARAM_KEYS:
        assert torch.equal(recovered[key], params0[key]), key


def test_multi_timestep_recovery_is_exact():
    params0 = _make_params(6)
    recorder = HistoryRecorder()

    active1 = torch.tensor([False, True, False, True, False, False])
    params1, _ = _simulate_timestep(params0, active1, recorder, timestep=1, seed=1)

    active2 = torch.zeros(params1["means"].shape[0], dtype=torch.bool)
    active2[0] = True
    active2[-1] = True
    params2, _ = _simulate_timestep(params1, active2, recorder, timestep=2, seed=2)

    rec1 = recover_state(params2, recorder.records, target_time=1)
    rec0 = recover_state(params2, recorder.records, target_time=0)
    rec2 = recover_state(params2, recorder.records, target_time=2)
    for key in PARAM_KEYS:
        assert torch.equal(rec1[key], params1[key]), key
        assert torch.equal(rec0[key], params0[key]), key
        assert torch.equal(rec2[key], params2[key]), key


def test_recover_state_validates_target_range():
    import pytest

    params0 = _make_params(4)
    recorder = HistoryRecorder()
    active = torch.tensor([True, False, False, True])
    params1, _ = _simulate_timestep(params0, active, recorder, timestep=1, seed=3)

    with pytest.raises(ValueError):
        recover_state(params1, recorder.records, target_time=-1)
    with pytest.raises(ValueError):
        recover_state(params1, recorder.records, target_time=2)


def test_records_save_and_load_roundtrip(tmp_path):
    params0 = _make_params(5)
    recorder = HistoryRecorder()
    active = torch.tensor([True, True, False, False, False])
    params1, _ = _simulate_timestep(params0, active, recorder, timestep=1, seed=4)

    recorder.save(tmp_path)
    loaded = HistoryRecorder.load(tmp_path)

    assert len(loaded) == 1
    recovered = revert_timestep(params1, loaded[0])
    for key in PARAM_KEYS:
        assert torch.equal(recovered[key], params0[key]), key


def test_revert_detects_violated_invariant():
    import pytest

    params0 = _make_params(4)
    recorder = HistoryRecorder()
    active = torch.tensor([True, False, False, False])
    params1, _ = _simulate_timestep(params0, active, recorder, timestep=1, seed=5)

    # Drop an inactive row — the preserved-scene invariant is broken and
    # recovery must fail loudly rather than return a wrong scene.
    corrupted = {k: v[1:] for k, v in params1.items()}
    recorder.records[0].active_end_mask = recorder.records[0].active_end_mask[1:]
    with pytest.raises(AssertionError):
        revert_timestep(corrupted, recorder.records[0])
