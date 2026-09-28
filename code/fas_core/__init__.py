"""Core numerical routines for Functional Ancestry Simplex (FAS).

The model-specific inference layer is intentionally separated from the ancestry
mathematics so the same code can be reused for LLM and diffusion experiments.
"""
from .active_probe import greedy_logdet_select, information_matrix
from .decompose import fas_decompose
from .conformal import conformal_pvalue
from .selective import signal_pvalue, signal_score, selective_decision
from .bootstrap import bootstrap_fas
