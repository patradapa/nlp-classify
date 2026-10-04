import sys
from pathlib import Path

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 2-train-test-model/ (so classical_ml is importable)

from classical_ml.preprocessing import (
    X_train_tfidf, X_test_tfidf, y_train, y_test,
)

print("[Track A][1/4] เทรน SVM (linear kernel) ...")
svm = SVC(kernel="linear", probability=True, random_state=42)
svm.fit(X_train_tfidf, y_train)

print("[Track A][2/4] ทำนาย SVM บน test set ...")
svm_preds = svm.predict(X_test_tfidf)
print("SVM:\n", classification_report(y_test, svm_preds))

print("[Track A][3/4] เทรน Random Forest (200 trees) ...")
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_tfidf, y_train)

print("[Track A][4/4] ทำนาย Random Forest บน test set ...")
rf_preds = rf.predict(X_test_tfidf)
print("Random Forest:\n", classification_report(y_test, rf_preds))
