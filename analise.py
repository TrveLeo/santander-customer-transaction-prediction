import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)

SEED = 42
OUT = Path("analise_output")
OUT.mkdir(exist_ok=True)

# ── Dados ────────────────────────────────────────────────────────────────────
print("Carregando dados e modelos...")
train  = pd.read_csv("data/train.csv")
FEATURES = [f"var_{i}" for i in range(200)]
X = train[FEATURES].values
y = train["target"].values

scaler = pickle.load(open("models/scaler.pkl", "rb"))
X_sc   = scaler.transform(X)

rf  = pickle.load(open("models/random_forest.pkl", "rb"))
nb  = pickle.load(open("models/naive_bayes.pkl",   "rb"))
svm = pickle.load(open("models/svm.pkl",           "rb"))

# Subamostra SVM (mesma do treino)
rng = np.random.RandomState(SEED)
idx_pos = np.where(y == 1)[0]
idx_neg = np.where(y == 0)[0]
n_pos   = int(20_000 * y.mean())
n_neg   = 20_000 - n_pos
idx_sub = np.concatenate([
    rng.choice(idx_pos, n_pos, replace=False),
    rng.choice(idx_neg, n_neg, replace=False),
])
rng.shuffle(idx_sub)
X_sub = X_sc[idx_sub]
y_sub = y[idx_sub]

# ── Probabilidades OOF via CV ────────────────────────────────────────────────
print("Gerando probabilidades OOF...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def oof_proba(modelo, Xd, yd, usa_proba=True):
    oof = np.zeros(len(yd))
    for idx_tr, idx_val in skf.split(Xd, yd):
        modelo.fit(Xd[idx_tr], yd[idx_tr])
        if usa_proba:
            oof[idx_val] = modelo.predict_proba(Xd[idx_val])[:, 1]
        else:
            oof[idx_val] = modelo.decision_function(Xd[idx_val])
    return oof

prob_rf  = oof_proba(RandomForestClassifier(n_estimators=200, max_features="sqrt",
                     min_samples_leaf=5, class_weight="balanced",
                     n_jobs=-1, random_state=SEED), X_sc, y)
prob_nb  = oof_proba(GaussianNB(), X_sc, y)
prob_svm = oof_proba(SVC(kernel="rbf", C=1.0, gamma="scale",
                     class_weight="balanced", probability=True,
                     random_state=SEED), X_sub, y_sub)

modelos = {
    "Random Forest": (prob_rf,  y),
    "Naive Bayes":   (prob_nb,  y),
    "SVM (sub 20k)": (prob_svm, y_sub),
}

# ── 1. Curvas ROC ─────────────────────────────────────────────────────────────
print("Plotando curvas ROC...")
fig, ax = plt.subplots(figsize=(7, 6))
cores = {"Random Forest": "steelblue", "Naive Bayes": "tomato", "SVM (sub 20k)": "seagreen"}
for nome, (prob, ytrue) in modelos.items():
    fpr, tpr, _ = roc_curve(ytrue, prob)
    auc = roc_auc_score(ytrue, prob)
    ax.plot(fpr, tpr, label=f"{nome} (AUC={auc:.4f})", color=cores[nome], lw=2)
ax.plot([0,1],[0,1], "k--", lw=1)
ax.set_xlabel("Taxa de Falsos Positivos")
ax.set_ylabel("Taxa de Verdadeiros Positivos")
ax.set_title("Curvas ROC — Comparação dos Modelos")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "roc_curves.png", dpi=150)
plt.close()
print("  Salvo: roc_curves.png")

# ── 2. Curvas Precisão-Revocação ─────────────────────────────────────────────
print("Plotando curvas Precisão-Revocação...")
fig, ax = plt.subplots(figsize=(7, 6))
for nome, (prob, ytrue) in modelos.items():
    prec, rec, _ = precision_recall_curve(ytrue, prob)
    ap = average_precision_score(ytrue, prob)
    ax.plot(rec, prec, label=f"{nome} (AP={ap:.4f})", color=cores[nome], lw=2)
baseline = y.mean()
ax.axhline(baseline, color="gray", ls="--", lw=1, label=f"Baseline ({baseline:.2f})")
ax.set_xlabel("Revocação")
ax.set_ylabel("Precisão")
ax.set_title("Curvas Precisão-Revocação")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "precision_recall_curves.png", dpi=150)
plt.close()
print("  Salvo: precision_recall_curves.png")

# ── 3. Matrizes de confusão ──────────────────────────────────────────────────
print("Plotando matrizes de confusão...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (nome, (prob, ytrue)) in zip(axes, modelos.items()):
    threshold = 0.5
    ypred = (prob >= threshold).astype(int)
    cm = confusion_matrix(ytrue, ypred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Não (0)", "Sim (1)"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(nome)
plt.suptitle("Matrizes de Confusão (threshold=0.5)", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Salvo: confusion_matrices.png")

# ── 4. Importância de features (Random Forest) ───────────────────────────────
print("Plotando importância de features...")
imp = pd.read_csv("models/rf_feature_importance.csv", index_col=0, header=None)
imp.columns = ["importancia"]
imp = imp.sort_values("importancia", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 5))
imp["importancia"].plot.bar(ax=ax, color="steelblue")
ax.set_title("Top 20 Features Mais Importantes — Random Forest")
ax.set_ylabel("Importância (redução de impureza)")
ax.set_xlabel("Feature")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=150)
plt.close()
print("  Salvo: feature_importance.png")

# ── 5. Distribuição de probabilidades ────────────────────────────────────────
print("Plotando distribuição de probabilidades...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (nome, (prob, ytrue)) in zip(axes, modelos.items()):
    ax.hist(prob[ytrue==0], bins=50, alpha=0.6, density=True, label="target=0", color="steelblue")
    ax.hist(prob[ytrue==1], bins=50, alpha=0.6, density=True, label="target=1", color="tomato")
    ax.set_title(nome)
    ax.set_xlabel("Probabilidade prevista")
    ax.set_ylabel("Densidade")
    ax.legend()
plt.suptitle("Distribuição das Probabilidades Previstas por Classe")
plt.tight_layout()
plt.savefig(OUT / "prob_distributions.png", dpi=150)
plt.close()
print("  Salvo: prob_distributions.png")

# ── 6. Tabela resumo ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("ANÁLISE COMPARATIVA DOS MODELOS")
print("="*65)

rows = []
for nome, (prob, ytrue) in modelos.items():
    ypred = (prob >= 0.5).astype(int)
    rep   = classification_report(ytrue, ypred, output_dict=True)
    auc   = roc_auc_score(ytrue, prob)
    ap    = average_precision_score(ytrue, prob)
    rows.append({
        "Modelo":        nome,
        "AUC-ROC":       round(auc, 4),
        "AP":            round(ap, 4),
        "Acurácia":      round(rep["accuracy"], 4),
        "Precisão (1)":  round(rep["1"]["precision"], 4),
        "Revocação (1)": round(rep["1"]["recall"], 4),
        "F1 (1)":        round(rep["1"]["f1-score"], 4),
    })

df_res = pd.DataFrame(rows).set_index("Modelo")
print(df_res.to_string())
df_res.to_csv(OUT / "resumo_comparativo.csv")
print(f"\n  Salvo: resumo_comparativo.csv")

print(f"""
INTERPRETAÇÃO:
- AUC-ROC: métrica principal do desafio — quanto mais próximo de 1, melhor
- AP (Average Precision): resume a curva Precisão-Revocação — importante para dados desbalanceados
- Precisão (1): dos previstos como "Sim", quantos eram realmente "Sim"
- Revocação (1): dos que eram realmente "Sim", quantos o modelo encontrou
- Nota: AUC do SVM avaliado em subamostra de 20k — não comparável diretamente
""")
print(f"Análise concluída. Resultados em: {OUT}/")
