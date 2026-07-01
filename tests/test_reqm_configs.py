from infusers import QM


def test_qm_validate() -> None:
    QM.validate()


def test_list_quant_recipes() -> None:
    names = QM.list_configs()
    assert "quant/flux/klein9b/image_basic" in names
    assert "quant/flux/klein9b/image_basic_offload" in names
    assert "quant/flux/klein9b/pano_basic" in names
    assert "quant/flux/klein9bkv/image_basic" in names
    assert "quant/flux/klein9bkv/image_basic_offload" in names
    assert "quant/image_basic_dummy" in names
    assert "models/flux/klein9b" in names
    assert "models/flux/klein9bkv" in names
