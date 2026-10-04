import pickle
from pathlib import Path

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


def build_contingency_table(y_true, preds_a, preds_b):
    correct_a = np.array(preds_a) == np.array(y_true)
    correct_b = np.array(preds_b) == np.array(y_true)
    both_correct = np.sum(correct_a & correct_b)
    only_a = np.sum(correct_a & ~correct_b)
    only_b = np.sum(~correct_a & correct_b)
    both_wrong = np.sum(~correct_a & ~correct_b)
    return [[both_correct, only_a], [only_b, both_wrong]]


if __name__ == "__main__":
    cache_path = Path("evaluation/preds_cache.pkl")
    if not cache_path.exists():
        raise SystemExit("รัน evaluation/evaluate_all.py ก่อนเพื่อสร้าง preds_cache.pkl")

    with open(cache_path, "rb") as f:
        cached = pickle.load(f)

    y_test = cached.pop("y_test")
    model_names = list(cached.keys())

    print(f"models available: {model_names}")

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_a, name_b = model_names[i], model_names[j]
            table = build_contingency_table(y_test, cached[name_a], cached[name_b])
            result = mcnemar(table, exact=True)
            print(f"{name_a} vs {name_b}: McNemar's test p-value = {result.pvalue:.4f}")
