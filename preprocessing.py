import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

SEED = 42
FEATURES = [f"var_{i}" for i in range(200)]
OUT = Path("models")
OUT.mkdir(exist_ok=True)


def carregar_dados():
    print("Carregando dados...")
    train = pd.read_csv("data/train.csv")
    test  = pd.read_csv("data/test.csv")
    print(f"  Treino : {train.shape}")
    print(f"  Teste  : {test.shape}")
    return train, test


def verificar_qualidade(train, test):
    print("\nVerificando qualidade dos dados...")

    # Valores ausentes
    miss_train = train[FEATURES].isnull().sum().sum()
    miss_test  = test[FEATURES].isnull().sum().sum()
    print(f"  Valores ausentes — treino: {miss_train} | teste: {miss_test}")

    # Distribuição do target
    vc = train["target"].value_counts()
    print(f"  Target=0: {vc[0]:,} ({vc[0]/len(train)*100:.1f}%)")
    print(f"  Target=1: {vc[1]:,} ({vc[1]/len(train)*100:.1f}%)")

    # Features constantes (mesmo valor para todas as amostras)
    constantes = [f for f in FEATURES if train[f].nunique() == 1]
    print(f"  Features constantes: {len(constantes)}")

    return {
        "missing_train": miss_train,
        "missing_test": miss_test,
        "desbalanceamento": vc[1] / len(train),
        "features_constantes": constantes,
    }


def padronizar(train, test):
    print("\nPadronizando features (StandardScaler)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURES])
    X_test  = scaler.transform(test[FEATURES])

    # Salva scaler para uso em analise.py e predict
    pickle.dump(scaler, open(OUT / "scaler.pkl", "wb"))
    print(f"  Scaler salvo em models/scaler.pkl")
    print(f"  Média pós-escala (amostra): {X_train.mean(axis=0)[:3].round(4)}")
    print(f"  Std  pós-escala (amostra): {X_train.std(axis=0)[:3].round(4)}")

    return X_train, X_test


def subamostra_svm(X, y, n=20_000):
    """
    Subamostra estratificada para viabilizar treino do SVM (custo O(n²)).
    Mantém proporção original de classes.
    """
    print(f"\nGerando subamostra estratificada para SVM (n={n:,})...")
    rng = np.random.RandomState(SEED)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_pos = int(n * y.mean())
    n_neg = n - n_pos
    idx_sub = np.concatenate([
        rng.choice(idx_pos, n_pos, replace=False),
        rng.choice(idx_neg, n_neg, replace=False),
    ])
    rng.shuffle(idx_sub)
    print(f"  Positivos: {n_pos:,} ({n_pos/n*100:.1f}%) | Negativos: {n_neg:,}")
    return X[idx_sub], y[idx_sub]


if __name__ == "__main__":
    train, test = carregar_dados()
    info = verificar_qualidade(train, test)

    y_train = train["target"].values
    X_train, X_test = padronizar(train, test)

    X_svm, y_svm = subamostra_svm(X_train, y_train)

    # Salva arrays prontos para uso em train.py
    np.save(OUT / "X_train.npy", X_train)
    np.save(OUT / "X_test.npy",  X_test)
    np.save(OUT / "y_train.npy", y_train)
    np.save(OUT / "X_svm.npy",   X_svm)
    np.save(OUT / "y_svm.npy",   y_svm)

    print("\nPré-processamento concluído. Arrays salvos em models/")
    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  X_svm   : {X_svm.shape}")
