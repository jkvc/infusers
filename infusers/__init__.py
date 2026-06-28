"""Custom inferencer logic for jkvc ML inference."""

from reqm import QuantManager

from infusers import configs

QM = QuantManager(configs)

__all__ = ["QM"]
