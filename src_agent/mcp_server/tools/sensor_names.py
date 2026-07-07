"""Case-insensitive resolution of sensor/feature names to their canonical
CMAPSS spelling (e.g. "ps30" -> "Ps30").

Tool-calling models sometimes lowercase parameter values even when the tool
description states the exact spelling. SQL column/identifier lookups are
case-insensitive in SQLite, but the Python-level lookups downstream of them
(dict keys, dataframe columns) are not, so a wrong-case name would otherwise
fail confusingly deep in a tool rather than resolving to what was obviously
meant. Genuinely unknown names are left untouched, so callers can still
report them as unknown rather than silently guessing.
"""

from src.const import CYCLE_COLUMN, SENSOR_NAMES

_CANONICAL_NAMES_BY_LOWER = {
    name.lower(): name for name in [CYCLE_COLUMN, *SENSOR_NAMES]
}


def resolve_feature_name(requested_name: str) -> str:
    """Map a possibly differently-cased name to its canonical spelling."""
    return _CANONICAL_NAMES_BY_LOWER.get(requested_name.lower(), requested_name)


def resolve_feature_names(requested_names: list[str]) -> list[str]:
    """Resolve each name in a list; order and unknown entries are preserved."""
    return [resolve_feature_name(name) for name in requested_names]
