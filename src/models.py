import os
from pathlib import Path
from typing import Dict, Union, Any
import numpy as np
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

class MNISTModelBench:
    """
    Classe responsável pela inicialização, treinamento idempotente, 
    persistência e carregamento dos modelos de Machine Learning (KNN, Random Forest e MLP).
    """
    def __init__(
        self, 
        models_dir: Union[str, Path] = "/content/drive/MyDrive/mini_projeto_mnist/models",
        random_state: int = 42
    ):
        """
        Inicializa o benchmark de modelos e garante a existência da pasta de artefatos.
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.models: Dict[str, Any] = {}

    def get_model_path(self, model_name: str) -> Path:
        """Retorna o caminho padronizado do arquivo .joblib do modelo."""
        filename = f"{model_name.lower().replace(' ', '_')}.joblib"
        return self.models_dir / filename

    def train_or_load_knn(
        self, X_train: np.ndarray, y_train: np.ndarray, n_neighbors: int = 3
    ) -> KNeighborsClassifier:
        """
        Treina ou carrega o classificador K-Nearest Neighbors (KNN).
        
        JUSTIFICATIVA TEÓRICA:
        - O KNN é um algoritmo não-paramétrico baseado em distância Euclidiana.
        - Definimos n_neighbors=3 para capturar fronteiras de decisão locais sem suavização excessiva.
        - Utiliza n_jobs=-1 para paralelizar a busca de vizinhos em múltiplos núcleos de CPU.
        """
        model_name = "knn"
        filepath = self.get_model_path(model_name)

        if filepath.exists():
            print(f"[CACHE HIT] Carregando modelo KNN de: {filepath}")
            model = joblib.load(filepath)
        else:
            print("[CACHE MISS] Treinando modelo KNN (K-Nearest Neighbors)...")
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", n_jobs=-1)
            model.fit(X_train, y_train)
            joblib.dump(model, filepath)
            print(f"[PERSISTÊNCIA] Modelo KNN salvo com sucesso em: {filepath}")

        self.models["KNN"] = model
        return model

    def train_or_load_rf(
        self, X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 100
    ) -> RandomForestClassifier:
        """
        Treina ou carrega o classificador Random Forest.
        
        JUSTIFICATIVA TEÓRICA:
        - Ensemble de Árvores de Decisão que reduz variância via Bagging e amostragem de atributos.
        - n_estimators=100 oferece um equilíbrio entre capacidade preditiva e tempo de processamento.
        - Robusto a ruídos e pequenas variações na intensidade dos pixels.
        """
        model_name = "random_forest"
        filepath = self.get_model_path(model_name)

        if filepath.exists():
            print(f"[CACHE HIT] Carregando modelo Random Forest de: {filepath}")
            model = joblib.load(filepath)
        else:
            print("[CACHE MISS] Treinando modelo Random Forest...")
            model = RandomForestClassifier(
                n_estimators=n_estimators, 
                random_state=self.random_state, 
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            joblib.dump(model, filepath)
            print(f"[PERSISTÊNCIA] Modelo Random Forest salvo com sucesso em: {filepath}")

        self.models["Random Forest"] = model
        return model

    def train_or_load_mlp(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> MLPClassifier:
        """
        Treina ou carrega a Redes Neural Perceptron Multicamadas (MLP).
        
        JUSTIFICATIVA TEÓRICA:
        - Arquitetura com duas camadas ocultas (128, 64 neurônios) e ativação ReLU.
        - Otimizador Adam com Early Stopping ativo para evitar overfitting e otimizar convergência.
        - Aprende representações não-lineares abstratas das imagens de dígitos.
        """
        model_name = "mlp"
        filepath = self.get_model_path(model_name)

        if filepath.exists():
            print(f"[CACHE HIT] Carregando modelo MLP de: {filepath}")
            model = joblib.load(filepath)
        else:
            print("[CACHE MISS] Treinando modelo MLP (Multi-Layer Perceptron)...")
            model = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                batch_size=256,
                max_iter=60,
                early_stopping=True,
                random_state=self.random_state,
                verbose=False
            )
            model.fit(X_train, y_train)
            joblib.dump(model, filepath)
            print(f"[PERSISTÊNCIA] Modelo MLP salvo com sucesso em: {filepath}")

        self.models["MLP"] = model
        return model

    def train_or_load_all(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Treina ou carrega sequencialmente os 3 modelos do projeto."""
        self.train_or_load_knn(X_train, y_train)
        self.train_or_load_rf(X_train, y_train)
        self.train_or_load_mlp(X_train, y_train)
        return self.models
