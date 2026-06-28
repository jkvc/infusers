"""Tests for reqm.instantiate_cached model sharing via klein9b.yaml."""

from __future__ import annotations

import pytest
from hydra.utils import instantiate
from reqm import clear_instantiate_cache

from infusers import QM


@pytest.fixture(autouse=True)
def _clear_model_cache() -> None:
    clear_instantiate_cache()


def test_identical_quant_builds_share_model(monkeypatch: pytest.MonkeyPatch) -> None:
    instances: list[object] = []

    class FakeKlein:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            instances.append(self)

    monkeypatch.setattr("infusers.model.klein.KleinModel", FakeKlein)

    q1 = QM.build("quant/flux/klein9b/image_basic")
    q2 = QM.build("quant/flux/klein9b/image_basic")

    assert len(instances) == 1
    assert q1.model is q2.model


def test_offload_model_config_builds_separate_cached_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[object] = []

    class FakeKlein:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            instances.append(self)

    monkeypatch.setattr("infusers.model.klein.KleinModel", FakeKlein)

    basic_cfg = QM.get_config("quant/flux/klein9b/image_basic")
    offload_cfg = QM.get_config("quant/flux/klein9b/image_basic_offload")

    basic_model = instantiate(basic_cfg.model)
    offload_model = instantiate(offload_cfg.model)

    assert len(instances) == 2
    assert basic_model is not offload_model
    assert offload_model.kwargs["load_flow_on_cpu"] is True
