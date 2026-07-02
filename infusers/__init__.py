"""Custom ML inference — shared models via reqm, pluggable quants."""

from reqm import QuantManager

from infusers import configs

QM = QuantManager(configs)

__all__ = ["QM"]
