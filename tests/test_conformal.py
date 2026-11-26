import numpy as np
import pytest

from hallucination_lens.conformal import ConformalPredictor, conformal_quantile, lac_scores


def test_quantile_uses_finite_sample_correction():
    scores = np.linspace(0.0, 1.0, 22)
    # report, Sec. 3.6: n = 22, alpha = 0.05 -> level 22/22 -> threshold is the largest score
    assert conformal_quantile(scores, 0.05) == 1.0
    # worked example from docs/HOW_IT_WORKS.md
    toy = [0.05, 0.10, 0.12, 0.20, 0.25, 0.31, 0.40, 0.48, 0.55, 0.90]
    assert conformal_quantile(toy, 0.2) == 0.55


def test_too_few_points_means_everything_is_included():
    assert conformal_quantile(np.linspace(0, 1, 10), 0.05) == np.inf


def test_lac_scores():
    proba = np.array([[0.9, 0.1], [0.3, 0.7]])
    assert lac_scores(proba, np.array([0, 1])) == pytest.approx([0.1, 0.3])


def test_worked_example_sets():
    toy = np.array([0.05, 0.10, 0.12, 0.20, 0.25, 0.31, 0.40, 0.48, 0.55, 0.90])
    # build calibration probabilities whose true-class surprise equals `toy`
    proba_cal = np.column_stack([1 - toy, toy])  # P(class 0) = 1 - s ; true label 0
    cp = ConformalPredictor(alpha=0.2).calibrate(proba_cal, np.zeros(10, dtype=int))
    assert cp.q_hat == pytest.approx(0.55)
    sets = cp.predict_set(np.array([[0.45, 0.55], [0.90, 0.10], [0.30, 0.70]]))
    assert sets.tolist() == [[True, True], [True, False], [False, True]]


def _simulate(rng, n, miscalibrated=True):
    """A deliberately over-confident classifier on a noisy problem."""
    y = rng.binomial(1, 0.65, n)
    logit = (2 * y - 1) * 0.8 + rng.normal(scale=1.0, size=n)
    p1 = 1 / (1 + np.exp(-(3.0 if miscalibrated else 1.0) * logit))
    return np.column_stack([1 - p1, p1]), y


def test_marginal_and_class_conditional_coverage_hold_on_average():
    rng = np.random.default_rng(0)
    alpha = 0.1
    cov, cov_c0, cov_c1 = [], [], []
    for _ in range(300):
        p_cal, y_cal = _simulate(rng, 150)
        p_te, y_te = _simulate(rng, 600)
        marg = ConformalPredictor(alpha, False).calibrate(p_cal, y_cal)
        cc = ConformalPredictor(alpha, True).calibrate(p_cal, y_cal)
        cov.append(marg.predict_set(p_te)[np.arange(600), y_te].mean())
        covered = cc.predict_set(p_te)[np.arange(600), y_te]
        cov_c0.append(covered[y_te == 0].mean())
        cov_c1.append(covered[y_te == 1].mean())
    assert np.mean(cov) >= 1 - alpha - 0.01
    assert np.mean(cov_c0) >= 1 - alpha - 0.015  # holds per class, not just overall
    assert np.mean(cov_c1) >= 1 - alpha - 0.015


def test_roundtrip_serialisation():
    rng = np.random.default_rng(1)
    p, y = _simulate(rng, 100)
    cp = ConformalPredictor(0.05, True).calibrate(p, y)
    clone = ConformalPredictor.from_dict(cp.to_dict())
    assert np.array_equal(cp.predict_set(p), clone.predict_set(p))


def test_predict_before_calibrate_fails():
    with pytest.raises(RuntimeError):
        ConformalPredictor().predict_set(np.array([[0.5, 0.5]]))


def test_matches_mapie_lac():
    """Cross-check the from-scratch implementation against the MAPIE library."""
    mapie = pytest.importorskip("mapie.classification")
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(3)
    X = rng.normal(size=(900, 4))
    y = (X[:, 0] + 0.8 * rng.normal(size=900) > 0).astype(int)
    X_tr, X_cal, X_te = X[:300], X[300:600], X[600:]
    y_tr, y_cal = y[:300], y[300:600]
    clf = LogisticRegression().fit(X_tr, y_tr)
    for alpha in (0.05, 0.2):
        m = mapie.SplitConformalClassifier(
            estimator=clf, confidence_level=1 - alpha, conformity_score="lac", prefit=True
        )
        m.conformalize(X_cal, y_cal)
        _, theirs = m.predict_set(X_te)
        ours = ConformalPredictor(alpha).calibrate(clf.predict_proba(X_cal), y_cal).predict_set(
            clf.predict_proba(X_te)
        )
        assert np.array_equal(theirs[:, :, 0], ours)
