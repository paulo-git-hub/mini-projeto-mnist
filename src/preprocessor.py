import os
from pathlib import Path
from typing import Tuple, Union, Dict
import numpy as np
from sklearn.model_selection import train_test_split

class MNISTPreprocessor:
    """
    Classe responsável pelo pipeline de pré-processamento, divisão estratificada
    e normalização do dataset MNIST com suporte a persistência idempotente.
    """
    def __init__(
        self,
        data_dir: Union[str, Path] = "/content/drive/MyDrive/mini_projeto_mnist/data",
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ):
        """
        Inicializa o pré-processador configurando proporções e diretórios.

        JUSTIFICATIVA TEÓRICA E ENGENHARIA:
        1. Divisão Estratificada (stratify=y):
           Garante que a proporção exata de cada dígito (0 a 9) seja preservada
           identicamente nos conjuntos de Treino, Validação e Teste. Isso impede
           vieses de amostragem que comprometeriam a generalização do modelo.

        2. Normalização Min-Max para [0.0, 1.0]:
           Os pixels originais possuem escala de cinza inteira em [0, 255]. A conversão
           para ponto flutuante dividindo por 255.0 é indispensável para:
           - KNN: Evita que atributos com maiores magnitudes dominem o cálculo da
             distância Euclidiana.
           - Redes Neurais (MLP): Garante a estabilidade numérica da retropropagação,
             otimizando a velocidade de convergência do Gradiente Descendente e
             prevenindo explosão/desaparecimento de gradientes.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_path = self.data_dir / "processed_data.npz"
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def process_and_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Executa a normalização dos pixels e as partições estratificadas.
        Aplica idempotência estrita: recarrega os arrays de processed_data.npz se já existirem.
        """
        if self.processed_path.exists():
            print(f"[CACHE HIT] Partições pré-processadas carregadas de: {self.processed_path}")
            data = np.load(self.processed_path)
            return {
                "X_train": data["X_train"], "y_train": data["y_train"],
                "X_val": data["X_val"], "y_val": data["y_val"],
                "X_test": data["X_test"], "y_test": data["y_test"]
            }

        print("[CACHE MISS] Executando escalamento Min-Max e divisão estratificada...")

        # 1. Normalização de atributos para o intervalo [0.0, 1.0]
        X_normalized = X.astype(np.float32) / 255.0

        # 2. Primeira partição estratificada: Separação do Teste final (15%)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_normalized,
            y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state
        )

        # 3. Segunda partição estratificada: Separação de Treino e Validação (15% do total)
        relative_val_size = self.val_size / (1.0 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=relative_val_size,
            stratify=y_train_val,
            random_state=self.random_state
        )

        # 4. Persistência compactada para garantir idempotência em disco
        np.savez_compressed(
            self.processed_path,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test
        )
        print(f"[PERSISTÊNCIA] Partições salvas com sucesso em: {self.processed_path}")

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test
        }
