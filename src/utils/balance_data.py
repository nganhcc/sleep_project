from imblearn.over_sampling import SMOTE


def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)
    return X_bal, y_bal
