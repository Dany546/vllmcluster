import importlib.util
import os


def _load_local_utils():
    fp = os.path.join(os.path.dirname(__file__), '..', 'utils.py')
    fp = os.path.abspath(fp)
    spec = importlib.util.spec_from_file_location('vllmcluster_utils', fp)
    vll_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vll_utils)
    return vll_utils


def test_color_for_group_deterministic():
    v = _load_local_utils()
    a = v.color_for_group('resnet50')
    b = v.color_for_group('resnet50')
    assert a == b


def test_parse_model_group():
    v = _load_local_utils()
    assert v.parse_model_group('resnet50_l2') == ('resnet50', 'l2')
    assert v.parse_model_group('clip') == ('clip', None)
