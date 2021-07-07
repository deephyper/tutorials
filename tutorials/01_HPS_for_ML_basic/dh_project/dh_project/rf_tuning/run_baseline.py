def run_baseline():
    import numpy as np
    from sklearn.utils import check_random_state
    from sklearn.ensemble import RandomForestClassifier
    from deephyper.benchmark.datasets import airlines as dataset

    rs_data = np.random.RandomState(seed=42)

    ratio_test = 0.33
    ratio_valid = (1 - ratio_test) * 0.33

    train, valid, test, _ = dataset.load_data(
        random_state=rs_data,
        test_size=ratio_test,
        valid_size=ratio_valid,
        categoricals_to_integers=True,
    )

    rs_classifier = check_random_state(42)

    classifier = RandomForestClassifier(n_jobs=4, random_state=rs_classifier)
    classifier.fit(*train)

    acc_train = classifier.score(*train)
    acc_valid = classifier.score(*valid)
    acc_test = classifier.score(*test)

    print(f"Accuracy on Training: {acc_train:.3f}")
    print(f"Accuracy on Validation: {acc_valid:.3f}")
    print(f"Accuracy on Testing: {acc_test:.3f}")


if __name__ == "__main__":
    run_baseline()