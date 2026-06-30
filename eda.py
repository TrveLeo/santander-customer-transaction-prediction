import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUT = Path("eda_output")
OUT.mkdir(exist_ok=True)

print("Carregando dados...")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

FEATURES = [f"var_{i}" for i in range(200)]

# ── 1. Info geral ────────────────────────────────────────────────────────────
print("\n=== SHAPE ===")
print(f"Treino : {train.shape}")
print(f"Teste  : {test.shape}")

print("\n=== TARGET ===")
vc = train["target"].value_counts()
print(vc)
print(f"% positivo: {vc[1]/len(train)*100:.2f}%")

print("\n=== MISSING ===")
miss_train = train.isnull().sum().sum()
miss_test  = test.isnull().sum().sum()
print(f"Treino: {miss_train} | Teste: {miss_test}")

# ── 2. Estatísticas descritivas ──────────────────────────────────────────────
print("\nCalculando estatísticas...")
stats = pd.DataFrame({
    "mean_train":  train[FEATURES].mean(),
    "std_train":   train[FEATURES].std(),
    "mean_test":   test[FEATURES].mean(),
    "std_test":    test[FEATURES].std(),
})
stats["mean_diff"] = (stats["mean_train"] - stats["mean_test"]).abs()
stats["std_diff"]  = (stats["std_train"]  - stats["std_test"]).abs()
stats.to_csv(OUT / "stats_comparison.csv")
print(f"Top 10 features com maior diferença de média (treino vs teste):")
print(stats.nlargest(10, "mean_diff")[["mean_train","mean_test","mean_diff"]])

# ── 3. Distribuição do target por feature (média) ────────────────────────────
print("\nCorrelação features × target...")
pos = train[train.target == 1][FEATURES].mean()
neg = train[train.target == 0][FEATURES].mean()
sep = (pos - neg).abs().sort_values(ascending=False)
sep.to_csv(OUT / "feature_separation.csv")
print("Top 10 features mais separáveis por média:")
print(sep.head(10))

# ── 4. Detecção de padrão sintético (value counts no teste) ─────────────────
print("\nAnalisando duplicatas sintéticas no teste...")
# Competição 2019: valores no teste foram duplicados — contagem de aparições é informativa
sample_vars = FEATURES[:20]
test_unique_ratio = {}
for v in sample_vars:
    n_unique = test[v].nunique()
    n_total  = len(test)
    test_unique_ratio[v] = n_unique / n_total
ratio_series = pd.Series(test_unique_ratio).sort_values()
print("Razão unique/total no teste (primeiras 20 vars):")
print(ratio_series)

# ── 5. Plots ────────────────────────────────────────────────────────────────
print("\nGerando plots...")

# 5a. Target distribution
fig, ax = plt.subplots(figsize=(5, 4))
vc.plot.bar(ax=ax, color=["steelblue", "tomato"])
ax.set_title("Distribuição do Target")
ax.set_xlabel("Target")
ax.set_ylabel("Contagem")
ax.set_xticklabels(["0 (não)", "1 (sim)"], rotation=0)
for p in ax.patches:
    ax.annotate(f"{p.get_height():,.0f}", (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "target_distribution.png", dpi=120)
plt.close()

# 5b. Top 16 features: distribuição pos vs neg
top_features = sep.head(16).index.tolist()
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
for ax, feat in zip(axes.flatten(), top_features):
    ax.hist(train[train.target==0][feat], bins=50, alpha=0.6, label="target=0", density=True, color="steelblue")
    ax.hist(train[train.target==1][feat], bins=50, alpha=0.6, label="target=1", density=True, color="tomato")
    ax.set_title(feat, fontsize=9)
    ax.legend(fontsize=7)
plt.suptitle("Top 16 Features: distribuição por classe", y=1.01)
plt.tight_layout()
plt.savefig(OUT / "top_features_distribution.png", dpi=120, bbox_inches="tight")
plt.close()

# 5c. Treino vs Teste: diferença de média
fig, ax = plt.subplots(figsize=(12, 4))
stats["mean_diff"].sort_values(ascending=False).head(30).plot.bar(ax=ax, color="mediumpurple")
ax.set_title("Top 30 features: diferença de média treino vs teste")
ax.set_ylabel("|mean_train - mean_test|")
plt.tight_layout()
plt.savefig(OUT / "train_test_mean_diff.png", dpi=120)
plt.close()

# 5d. Feature separation score
fig, ax = plt.subplots(figsize=(12, 4))
sep.head(30).plot.bar(ax=ax, color="darkorange")
ax.set_title("Top 30 features: separação por média (pos - neg)")
ax.set_ylabel("|mean_pos - mean_neg|")
plt.tight_layout()
plt.savefig(OUT / "feature_separation.png", dpi=120)
plt.close()

print(f"\nEDA concluída. Resultados em: {OUT}/")
print("  target_distribution.png")
print("  top_features_distribution.png")
print("  train_test_mean_diff.png")
print("  feature_separation.png")
print("  stats_comparison.csv")
print("  feature_separation.csv")
