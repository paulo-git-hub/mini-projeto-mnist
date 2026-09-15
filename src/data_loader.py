import os
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

class MNISTDataLoader:
    """
    Classe responsável pelo carregamento idempotente, persistência em disco local
    e análise exploratória (EDA) estatística e visual do dataset MNIST.
    """
    def __init__(
        self,
        data_dir: Union[str, Path] = "/content/drive/MyDrive/mini_projeto_mnist/data"
    ):
        """
        Inicializa o Loader garantindo a infraestrutura de diretórios em disco.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.data_dir / "mnist.npz"

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carrega o dataset MNIST aplicando idempotência estrita.
        Se o arquivo mnist.npz existir no disco local (Google Drive), efetua a leitura direta.
        Caso contrário, faz o download via OpenML, converte rótulos e salva de forma compactada.
        """
        if self.file_path.exists():
            print(f"[CACHE HIT] Dataset carregado do disco: {self.file_path}")
            data = np.load(self.file_path)
            return data["X"], data["y"]

        print("[CACHE MISS] Baixando dataset MNIST via OpenML (executado apenas na 1ª vez)...")
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
        y = y.astype(np.int64)

        # Persistência em disco compactada (.npz) para otimizar E/S no Google Drive
        np.savez_compressed(self.file_path, X=X, y=y)
        print(f"[PERSISTÊNCIA] Dados salvos com sucesso em: {self.file_path}")

        return X, y

    def analyze_distribution(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Gera relatório descritivo de dimensão, tipos de dados e tabela de frequência
        absoluta e relativa de cada classe (dígitos 0 a 9).
        """
        print("\n=== RELATÓRIO DE ESTRUTURA DOS DADOS ===")
        print(f"Dimensões: Matriz de Entradas X={X.shape} | Vetor de Rótulos y={y.shape}")
        print(f"Tipo dos Dados: {X.dtype} | Intervalo Numérico dos Pixels: [{X.min()}, {X.max()}]")
        print("=========================================\n")

        classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        percentages = (counts / total_samples) * 100

        df_balance = pd.DataFrame({
            "Classe (Dígito)": classes,
            "Contagem Absoluta": counts,
            "Proporção (%)": np.round(percentages, 2)
        })

        return df_balance

    def plot_sample_grid(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Gera e exibe uma grade visual 2x5 contendo a primeira amostra representativa
        de cada classe numérica (0 a 9) com formatação limpa.
        """
        fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
        axes = axes.ravel()

        for digit in range(10):
            # Obtém o primeiro índice onde o rótulo é igual ao dígito atual
            idx = np.where(y == digit)[0][0]
            img = X[idx].reshape(28, 28)

            axes[digit].imshow(img, cmap="gray")
            axes[digit].set_title(f"Dígito: {digit}", fontsize=10, fontweight="bold")
            axes[digit].axis("off")

        plt.suptitle("Amostras Representativas do Dataset MNIST (0 a 9)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()
