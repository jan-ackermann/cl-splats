"""Compact per-timestep history recording and exact scene-state recovery.

CL-Splats only optimises the *active* subset of Gaussians at each
continual-learning timestep: inactive Gaussians are never modified,
removed, or reordered relative to each other. Any past scene state is
therefore exactly recoverable from the **final model** plus a small
per-timestep delta:

* ``pre_params`` — the raw parameter rows of the Gaussians selected as
  active at the *start* of the timestep, i.e. their end-of-previous-
  timestep state.
* ``active_start_mask`` — which rows of the previous array were active.
* ``active_end_mask`` — which rows of the resulting array belong to the
  active lineage at the *end* of the timestep (densified children inherit
  the ``cl_active`` flag through gsplat's ops).

Reverting one timestep: the inactive rows of the current array are, in
order, exactly the ``~active_start_mask`` rows of the previous array
(every densification/pruning op preserves the relative order of
survivors); the active rows are restored from ``pre_params``. Applying
this recursively recovers the array of any earlier timestep bit-exactly,
including its row order.

Usage::

    cl-splats-history \\
        --ply outputs/gaussians_time_0002.ply \\
        --history-dir outputs/history \\
        --time 0 \\
        --out outputs/gaussians_recovered_t0.ply
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import typer
from loguru import logger

# Raw (pre-activation) parameter keys, in the layout CLGaussians stores them:
# means (N,3), scales log (N,3), quats (N,4), opacities logit (N,),
# sh0 (N,1,3), shN (N,K-1,3).
PARAM_KEYS = ("means", "scales", "quats", "opacities", "sh0", "shN")

app = typer.Typer(pretty_exceptions_show_locals=False)


@dataclasses.dataclass
class TimestepRecord:
    """Delta needed to revert one continual-learning timestep."""

    timestep: int
    active_start_mask: torch.Tensor  # bool over the start-of-timestep array
    pre_params: Dict[str, torch.Tensor]  # raw rows of the then-active Gaussians
    active_end_mask: Optional[torch.Tensor] = None  # bool over the end-of-timestep array


class HistoryRecorder:
    """Captures per-timestep deltas during training."""

    def __init__(self) -> None:
        self.records: List[TimestepRecord] = []

    def begin_timestep(
        self,
        timestep: int,
        active_mask: torch.Tensor,
        strategy_params,
    ) -> None:
        """Snapshot the active rows before any optimisation of *timestep*."""
        mask = active_mask.detach().bool()
        pre = {
            key: strategy_params[key].detach()[mask].cpu().clone()
            for key in PARAM_KEYS
        }
        self.records.append(
            TimestepRecord(
                timestep=timestep,
                active_start_mask=mask.cpu().clone(),
                pre_params=pre,
            )
        )

    def end_timestep(self, active_mask: torch.Tensor) -> None:
        """Record which rows of the final array belong to the active lineage."""
        assert self.records, "end_timestep called before begin_timestep"
        record = self.records[-1]
        assert record.active_end_mask is None, "timestep already finalised"
        record.active_end_mask = active_mask.detach().bool().cpu().clone()

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for record in self.records:
            torch.save(dataclasses.asdict(record), out_dir / f"t{record.timestep:04d}.pt")

    @staticmethod
    def load(history_dir: str | Path) -> List[TimestepRecord]:
        history_dir = Path(history_dir)
        records = []
        for path in sorted(history_dir.glob("t*.pt")):
            data = torch.load(path, map_location="cpu", weights_only=True)
            records.append(TimestepRecord(**data))
        return records


def revert_timestep(
    params: Dict[str, torch.Tensor], record: TimestepRecord
) -> Dict[str, torch.Tensor]:
    """Reconstruct the end-of-previous-timestep array from *params*.

    ``params`` must be the raw parameter arrays at the end of
    ``record.timestep``.
    """
    assert record.active_end_mask is not None, (
        f"record for t={record.timestep} was never finalised"
    )
    end_mask = record.active_end_mask
    start_mask = record.active_start_mask
    inactive_now = ~end_mask

    n_prev = start_mask.shape[0]
    n_inactive = int(inactive_now.sum())
    assert n_inactive == int((~start_mask).sum()), (
        "inactive row count changed within a timestep — the preserved-scene "
        "invariant was violated and the history cannot be recovered exactly"
    )

    out: Dict[str, torch.Tensor] = {}
    for key in PARAM_KEYS:
        current = params[key]
        previous = torch.empty((n_prev,) + current.shape[1:], dtype=current.dtype)
        # Survivor order is preserved by every densification/pruning op, so
        # the inactive rows map back 1:1 in order.
        previous[~start_mask] = current[inactive_now]
        previous[start_mask] = record.pre_params[key]
        out[key] = previous
    return out


def recover_state(
    final_params: Dict[str, torch.Tensor],
    records: List[TimestepRecord],
    target_time: int,
) -> Dict[str, torch.Tensor]:
    """Recover the raw parameter arrays at the end of *target_time*.

    ``final_params`` must correspond to the end of the latest recorded
    timestep. Valid targets range from ``records[0].timestep - 1`` (the
    state before the first recorded update) to ``records[-1].timestep``.
    """
    if not records:
        return final_params
    ordered = sorted(records, key=lambda r: r.timestep)
    earliest = ordered[0].timestep - 1
    latest = ordered[-1].timestep
    if not earliest <= target_time <= latest:
        raise ValueError(
            f"target_time={target_time} outside recoverable range [{earliest}, {latest}]"
        )

    state = final_params
    for record in reversed(ordered):
        if record.timestep <= target_time:
            break
        state = revert_timestep(state, record)
    return state


# ---------------------------------------------------------------------------
# Raw PLY IO (no activation round-trips — exact recovery requires raw values)
# ---------------------------------------------------------------------------

def load_raw_ply(path: str | Path) -> Dict[str, torch.Tensor]:
    """Load a CL-Splats checkpoint keeping log-scales / logit-opacities raw."""
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"]
    means = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32)
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float32)
    opacities = np.asarray(v["opacity"]).astype(np.float32)
    sh0 = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32)[:, None, :]

    rest_cols = sorted(
        [p.name for p in v.properties if p.name.startswith("f_rest_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    if rest_cols:
        rest = np.stack([v[c] for c in rest_cols], axis=-1).astype(np.float32)
        shN = rest.reshape(means.shape[0], -1, 3)
    else:
        shN = np.zeros((means.shape[0], 0, 3), dtype=np.float32)

    return {
        "means": torch.from_numpy(means),
        "scales": torch.from_numpy(scales),
        "quats": torch.from_numpy(quats),
        "opacities": torch.from_numpy(opacities),
        "sh0": torch.from_numpy(sh0),
        "shN": torch.from_numpy(shN),
    }


def export_raw_ply(params: Dict[str, torch.Tensor], path: str | Path) -> None:
    """Write raw parameter arrays in the standard 3DGS PLY layout."""
    import gsplat.exporter as exporter

    exporter.export_splats(
        means=params["means"],
        scales=params["scales"],
        quats=params["quats"],
        opacities=params["opacities"],
        sh0=params["sh0"],
        shN=params["shN"],
        save_to=str(path),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@app.command()
def main(
    ply: str = typer.Option(
        ..., "--ply", "-p", help="Final .ply checkpoint (latest trained timestep)."
    ),
    history_dir: str = typer.Option(
        "outputs/history", "--history-dir", help="Directory with per-timestep history records."
    ),
    time: int = typer.Option(
        ..., "--time", "-t", help="Timestep to recover (end-of-timestep state)."
    ),
    out: str = typer.Option(
        ..., "--out", "-o", help="Where to write the recovered .ply checkpoint."
    ),
) -> None:
    """Recover the scene state at an earlier timestep from the final model.

    Requires training with ``history.log_history=true`` so the per-timestep
    deltas were saved alongside the checkpoints.
    """
    records = HistoryRecorder.load(history_dir)
    if not records:
        logger.error("No history records found in '{}'.", history_dir)
        raise SystemExit(1)

    logger.info(
        "Loaded {} record(s) covering t={}..{}.",
        len(records),
        records[0].timestep,
        records[-1].timestep,
    )
    final_params = load_raw_ply(ply)
    state = recover_state(final_params, records, time)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_raw_ply(state, out_path)
    logger.info(
        "Recovered t={} ({:,} Gaussians) → {}", time, state["means"].shape[0], out_path
    )


if __name__ == "__main__":
    app()
