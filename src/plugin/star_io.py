# vim: set expandtab shiftwidth=4 softtabstop=4:
"""Helpers over an ArtiaX ParticleList for connected-component workflows.

Phase 7: read `aisCcId` off each particle and group by that key. Writing back
to STAR uses ArtiaX's `save_particle_list`, which preserves the custom column
via `_data_keys` → `write_file` round-trip.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

CC_ID_KEY = "aisCcId"
_FALLBACK_KEY = 0  # shown as "ALL" when the STAR lacks the aisCcId column


def group_particles_by_cc(partlist) -> Dict[int, List]:
    """Return {cc_id: [particle_id, ...]} across the partlist's ParticleData.

    Particles missing the `aisCcId` field are bucketed under key 0, which the
    CC table renders as a single "ALL" row.
    """
    groups: Dict[int, List] = defaultdict(list)
    for pid, particle in partlist.data:
        try:
            cc = int(particle[CC_ID_KEY])
        except (KeyError, ValueError, TypeError):
            cc = _FALLBACK_KEY
        groups[cc].append(pid)
    return groups


def has_cc_column(partlist) -> bool:
    """True if the partlist's ParticleData was loaded with an aisCcId column."""
    try:
        return CC_ID_KEY in partlist.data._data_keys
    except Exception:
        return False
