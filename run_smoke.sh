#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
python -m pytest -q tests/test_core.py -k 'exact_nnls or conformal_bounds or dictionary_geometry_raw_and_unit or signal_gate_and_selective_states or weighted_decomposition_uses_one_geometry or functional_mixture_reference_respects_global_scaling or data_governance_overlap_and_mcq_helpers or partitioned_logdet_respects_groups'
python scripts/sanity_demo.py >/dev/null
python scripts/verify_public_release.py
