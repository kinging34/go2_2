"""Locate the existing APEX checkout used by the MuJoCo Warp port."""

from __future__ import annotations

import os
from pathlib import Path


def find_apex_root(explicit: str | Path | None = None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())
    elif os.environ.get("APEX_ROOT"):
        candidates.append(Path(os.environ["APEX_ROOT"]).expanduser())
    else:
        here = Path(__file__).resolve().parent
        for base in (Path.cwd(), here, here.parent):
            candidates.extend((base / "APEX", base / "apex_mjwarp" / "APEX", base))
        candidates.append(Path("/home/hoge/university/laboratory/go2_2/apex_mjwarp/APEX"))

    for candidate in candidates:
        root = candidate.resolve()
        if ((root / "legged_gym/envs/param_config.yaml").is_file()
                and (root / "rsl_rl/rsl_rl/runners/on_policy_runner.py").is_file()):
            return root
    locations = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Existing APEX checkout was not found. Run inside apex_mjwarp, "
        "or pass --apex-root /path/to/apex_mjwarp/APEX. Searched: " + locations
    )
