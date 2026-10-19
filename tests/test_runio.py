import pytest
from omegaconf import OmegaConf

from flyloop.runio import apply_sweep, rewrite_index_flag


def test_rewrite_index_flag():
    assert rewrite_index_flag(["s.py", "--index", "3", "a=1"]) == ["s.py", "index=3", "a=1"]
    assert rewrite_index_flag(["s.py", "--index=7"]) == ["s.py", "index=7"]
    assert rewrite_index_flag(["s.py", "a=1"]) == ["s.py", "a=1"]


def test_apply_sweep():
    cfg = OmegaConf.create({"index": 1, "stimulus": {"contrast": 1.0, "dt": 0.005},
                            "sweep_list": [{"stimulus": {"contrast": 0.5}},
                                           {"stimulus": {"contrast": 0.25}}]})
    out = apply_sweep(cfg)
    assert out.stimulus.contrast == 0.25 and out.stimulus.dt == 0.005
    cfg.index = 2
    with pytest.raises(IndexError):
        apply_sweep(cfg)
    assert apply_sweep(OmegaConf.create({"index": 0, "sweep_list": []})).index == 0
