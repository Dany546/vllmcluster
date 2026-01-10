import os
import numpy as np

from pipelines import FeaturePipeline, StandardScalerWrapper, SpearmanFeatureSelector, PLSExtractor
from db_utils import create_grid_db


def test_pipeline_smoke(tmp_path):
    rng = np.random.RandomState(0)
    X = rng.randn(100, 10)
    y = rng.randn(100)

    # create temp db
    db_path = str(tmp_path / 'grid_test.db')
    create_grid_db(db_path)

    # run small pipeline
    scaler = StandardScalerWrapper()
    Xs = scaler.fit_transform(X)

    fs = SpearmanFeatureSelector(0.05)
    Xs = fs.fit_transform(Xs, y)

    pls = PLSExtractor(n_components=2)
    Xt = pls.fit_transform(Xs, y)

    assert Xt.shape[1] == 2
