import numpy as np

from src.bootstrap_intervals import bootstrap_test_metrics


def make_synthetic_data():
    rng = np.random.default_rng(0)
    labels = rng.choice([0, 1], size=400, p=[0.75, 0.25])
    probabilities = np.where(
        labels == 1,
        rng.beta(5, 2, size=len(labels)),
        rng.beta(2, 5, size=len(labels)),
    )
    return labels, probabilities


def test_bootstrap_results_have_expected_metrics_and_fields():
    labels, probabilities = make_synthetic_data()

    results = bootstrap_test_metrics(
        labels,
        probabilities,
        threshold=0.5,
        n_resamples=200,
        random_state=10,
    )

    assert set(results) == {"roc_auc", "precision", "recall", "f1"}
    for values in results.values():
        assert set(values) == {
            "observed",
            "interval_low",
            "interval_high",
            "bootstrap_std",
        }
        assert values["interval_low"] <= values["observed"]
        assert values["observed"] <= values["interval_high"]


def test_bootstrap_results_repeat_with_the_same_random_state():
    labels, probabilities = make_synthetic_data()

    first = bootstrap_test_metrics(
        labels, probabilities, 0.5, n_resamples=200, random_state=10
    )
    second = bootstrap_test_metrics(
        labels, probabilities, 0.5, n_resamples=200, random_state=10
    )

    assert first == second


def test_bootstrap_interval_bounds_change_with_random_state():
    labels, probabilities = make_synthetic_data()

    first = bootstrap_test_metrics(
        labels, probabilities, 0.5, n_resamples=200, random_state=10
    )
    second = bootstrap_test_metrics(
        labels, probabilities, 0.5, n_resamples=200, random_state=11
    )

    first_bounds = {
        metric: (values["interval_low"], values["interval_high"])
        for metric, values in first.items()
    }
    second_bounds = {
        metric: (values["interval_low"], values["interval_high"])
        for metric, values in second.items()
    }
    assert first_bounds != second_bounds
