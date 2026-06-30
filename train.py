import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
OUT = Path("models")
OUT.mkdir(exist_ok=True)

# ── Dados (pré-processados por preprocessing.py) ─────────────────────────────
print("Carregando dados pré-processados...")
arrays = ["X_train", "X_test", "y_train", "X_svm", "y_svm"]
if not all((OUT / f"{a}.npy").exists() for a in arrays):
    raise FileNotFoundError("Execute preprocessing.py antes de train.py")

X        = np.load(OUT / "X_train.npy")
X_test   = np.load(OUT / "X_test.npy")
y        = np.load(OUT / "y_train.npy")
X_sub    = np.load(OUT / "X_svm.npy")
y_sub    = np.load(OUT / "y_svm.npy")

print(f"Treino: {X.shape} | Positivos: {int(y.sum())} ({y.mean()*100:.1f}%)")
print(f"SVM subamostra: {X_sub.shape}")

print(f"Treino: {X.shape} | Positivos: {y.sum()} ({y.mean()*100:.1f}%)")

# ── Pré-processamento ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled      = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
pickle.dump(scaler, open(OUT / "scaler.pkl", "wb"))

# ── Validação cruzada estratificada ─────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def avaliar_modelo(modelo, X_tr, y_tr, X_val, y_val, usa_proba=True):
    modelo.fit(X_tr, y_tr)
    y_prob = modelo.predict_proba(X_val)[:, 1] if usa_proba else modelo.decision_function(X_val)
    auc = roc_auc_score(y_val, y_prob)
    y_pred = modelo.predict(X_val)
    return auc, y_pred, y_prob

# ── 1. Random Forest ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MODELO 1: Random Forest")
print("="*60)

rf = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1,
    random_state=SEED,
    oob_score=True,
)

aucs_rf = []
for fold, (idx_tr, idx_val) in enumerate(skf.split(X, y)):
    auc, y_pred, y_prob = avaliar_modelo(rf, X[idx_tr], y[idx_tr], X[idx_val], y[idx_val])
    aucs_rf.append(auc)
    print(f"  Fold {fold+1}: AUC = {auc:.4f}")

print(f"\n  AUC médio (CV): {np.mean(aucs_rf):.4f} ± {np.std(aucs_rf):.4f}")
print(f"\n  Relatório (fold final):")
print(classification_report(y[idx_val], y_pred, target_names=["Não", "Sim"]))
print("  Matriz de confusão (fold final):")
print(confusion_matrix(y[idx_val], y_pred))

rf.fit(X, y)
pickle.dump(rf, open(OUT / "random_forest.pkl", "wb"))
pred_rf = rf.predict_proba(X_test)[:, 1]

FEATURES = [f"var_{i}" for i in range(200)]
importancias = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"\n  Top 10 features mais importantes:")
print(importancias.head(10).to_string())
importancias.to_csv(OUT / "rf_feature_importance.csv")

# ── 2. Naive Bayes ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MODELO 2: Naive Bayes (Gaussiano)")
print("="*60)

nb = GaussianNB()

aucs_nb = []
for fold, (idx_tr, idx_val) in enumerate(skf.split(X, y)):
    auc, y_pred, y_prob = avaliar_modelo(nb, X[idx_tr], y[idx_tr], X[idx_val], y[idx_val])
    aucs_nb.append(auc)
    print(f"  Fold {fold+1}: AUC = {auc:.4f}")

print(f"\n  AUC médio (CV): {np.mean(aucs_nb):.4f} ± {np.std(aucs_nb):.4f}")
print(f"\n  Relatório (fold final):")
print(classification_report(y[idx_val], y_pred, target_names=["Não", "Sim"]))
print("  Matriz de confusão (fold final):")
print(confusion_matrix(y[idx_val], y_pred))

nb.fit(X, y)
pickle.dump(nb, open(OUT / "naive_bayes.pkl", "wb"))
pred_nb = nb.predict_proba(X_test)[:, 1]

# ── 3. SVM (subamostra — pré-processada por preprocessing.py) ────────────────
print("\n" + "="*60)
print(f"MODELO 3: SVM (kernel RBF) — subamostra {len(y_sub):,} amostras")
print("="*60)
print("  (SVM tem custo O(n²) — inviável em 200k amostras completas)")

svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=SEED,
)

aucs_svm = []
for fold, (idx_tr, idx_val) in enumerate(skf.split(X_sub, y_sub)):
    auc, y_pred, y_prob = avaliar_modelo(svm, X_sub[idx_tr], y_sub[idx_tr], X_sub[idx_val], y_sub[idx_val])
    aucs_svm.append(auc)
    print(f"  Fold {fold+1}: AUC = {auc:.4f}")

print(f"\n  AUC médio (CV na subamostra): {np.mean(aucs_svm):.4f} ± {np.std(aucs_svm):.4f}")
print(f"\n  Relatório (fold final):")
print(classification_report(y_sub[idx_val], y_pred, target_names=["Não", "Sim"]))
print("  Matriz de confusão (fold final):")
print(confusion_matrix(y_sub[idx_val], y_pred))

svm.fit(X_sub, y_sub)
pickle.dump(svm, open(OUT / "svm.pkl", "wb"))
pred_svm = svm.predict_proba(X_test)[:, 1]

# ── Resumo comparativo ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESUMO COMPARATIVO")
print("="*60)
print(f"  Random Forest : AUC = {np.mean(aucs_rf):.4f} ± {np.std(aucs_rf):.4f}")
print(f"  Naive Bayes   : AUC = {np.mean(aucs_nb):.4f} ± {np.std(aucs_nb):.4f}")
print(f"  SVM (sub)     : AUC = {np.mean(aucs_svm):.4f} ± {np.std(aucs_svm):.4f}")

# ── Submissions individuais ──────────────────────────────────────────────────
sub_base = pd.read_csv("data/sample_submission.csv")
for nome, preds in [("random_forest", pred_rf), ("naive_bayes", pred_nb), ("svm", pred_svm)]:
    sub = sub_base.copy()
    sub["target"] = preds
    sub.to_csv(f"submission_{nome}.csv", index=False)
    print(f"\n  Submission salva: submission_{nome}.csv")

print("\nTreinamento concluído. Modelos salvos em models/")
