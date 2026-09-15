import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

class MNISTEvaluator:
    """
    Classe responsável pela avaliação comparativa holística de modelos de ML no dataset MNIST.
    Gera tabelas consolidadas de desempenho (Acurácia, Precisão, Recall, F1-Score) e
    matrizes de confusão 10x10 com visualização gráfica via Seaborn.
    """
    def __init__(self, class_names: list = [str(i) for i in range(10)]):
        self.class_names = class_names

    def compute_metrics(
        self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Calcula as métricas consolidadas (Acurácia, Precisão, Recall e F1-Score Macro)
        para cada modelo no conjunto de teste inédito.
        """
        records = []

        for name, model in models.items():
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="macro"
            )

            records.append({
                "Modelo": name,
                "Acurácia (%)": np.round(acc * 100, 2),
                "Precisão (Macro %)": np.round(precision * 100, 2),
                "Recall (Macro %)": np.round(recall * 100, 2),
                "F1-Score (Macro %)": np.round(f1 * 100, 2)
            })

        df_metrics = pd.DataFrame(records).sort_values(by="Acurácia (%)", ascending=False)
        return df_metrics.reset_index(drop=True)

    def plot_confusion_matrices(
        self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray
    ) -> None:
        """
        Gera e exibe matrizes de confusão 10x10 lado a lado para comparação direta
        dos padrões de erro entre KNN, Random Forest e MLP.
        """
        num_models = len(models)
        fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))

        if num_models == 1:
            axes = [axes]

        for idx, (name, model) in enumerate(models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=axes[idx]
            )

            axes[idx].set_title(f"Matriz de Confusão: {name}", fontsize=12, fontweight="bold")
            axes[idx].set_xlabel("Rótulo Preditivo", fontsize=10)
            axes[idx].set_ylabel("Rótulo Verdadeiro", fontsize=10)

        plt.suptitle("Comparativo de Matrizes de Confusão (10x10) - Teste", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.show()
