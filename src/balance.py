import numpy as np


def get_class_count(y):
    classes, counts = np.unique(y, return_counts=True)
    return dict(zip(classes, counts))


def oversampling(X, y):
    class_counts = get_class_count(y)
    max_count = max(class_counts.values())
    X_bal, y_bal = [], []
    for c in class_counts:
        X_c = X[y == c]
        y_c = y[y == c]

        n_to_add = max_count - len(y_c)
        if n_to_add > 0:
            idx = np.random.choice(len(X_c), n_to_add, replace=True)
            X_c_new = np.concatenate([X_c, X_c[idx]], axis=0)
            y_c_new = np.concatenate([y_c, y_c[idx]], axis=0)
        else:
            X_c_new, y_c_new = X_c, y_c
        X_bal.append(X_c_new)
        y_bal.append(y_c_new)

    X_bal = np.concatenate(X_bal)
    y_bal = np.concatenate(y_bal)

    # shuffle
    perm = np.random.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


def undersampling(X, y):
    class_counts = get_class_count(y)
    min_count = min(class_counts.values())
    X_bal, y_bal = [], []
    for c in class_counts:
        X_c = X[y == c]
        y_c = y[y == c]

        idx = np.random.choice(len(X_c), min_count, replace=False)
        X_c_new = X_c[idx]
        y_c_new = y_c[idx]

        X_bal.append(X_c_new)
        y_bal.append(y_c_new)
    X_bal = np.concatenate(X_bal)
    y_bal = np.concatenate(y_bal)

    perm = np.random.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


def smote(X, y, k=5):
    class_counts = get_class_count(y)
    max_count = max(class_counts.values())
    X_new, y_new = [X.copy()], [y.copy()]

    for c in class_counts:
        X_c, y_c = X[y == c], y[y == c]
        n_to_add = max_count - len(y_c)

        if n_to_add > 0:
            # pair wise distances inside c
            diff = X_c[:, np.newaxis, :] - X_c[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))

            neighbor_ids = np.argsort(dist, axis=1)[:, 1 : k + 1]

            synthetic_samples = []
            for _ in range(n_to_add):
                i = np.random.randint(0, len(X_c))
                nn = np.random.choice(neighbor_ids[i])
                lam = np.random.rand()
                new_sample = X_c[i] + lam * (X_c[nn] - X_c[i])
                synthetic_samples.append(new_sample)

            synthetic_samples = np.array(synthetic_samples)
            X_new.append(synthetic_samples)
            y_new.append(np.full(len(synthetic_samples), c))

    X_bal = np.concatenate(X_new)
    y_bal = np.concatenate(y_new)

    perm = np.random.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


def balance_classes(X, y, method="oversampling"):
    if method == "oversampling":
        return oversampling(X, y)
    elif method == "undersampling":
        return undersampling(X, y)
    elif method == "smote":
        return smote(X, y)
    else:
        raise ValueError("undentified method")
