from __future__ import annotations

import re


def derive_base_run_name(run_id: str) -> str:
    """Extract the user-given name from a run_id, stripping timestamp prefix and any _vN suffix."""
    match = re.match(r"^run_\d{4}-\d{2}-\d{2}_\d{4}(?:_(.+))?$", run_id)
    if not match:
        return ""
    name = match.group(1) or ""
    return re.sub(r"_v\d+$", "", name)
