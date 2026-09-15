import os
from pathlib import Path
from typing import Union, Tuple, Dict, List, Any
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import joblib

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

class MNISTOODBench:
    """
    Classe responsável pelos testes de estresse, degradação Out-Of-Distribution (OOD),
    mascaramento de classes (Class Masking) e padronização dinâmica de fundo via OpenCV.
    """
    def __init__(
        self,
        models_dir: Union[str, Path] = "/content/drive/MyDrive/mini_projeto_mnist/models",
        random_state: int = 42
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

    def train_or_load_masked_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hidden_classes: List[int] = [4, 7]
    ) -> MLPClassifier:
        """
        Treina ou carrega do cache um modelo MLP privado de duas classes (Desafio A).
        """
        filepath = self.models_dir / "mlp_masked_4_7.joblib"

        if filepath.exists():
            return joblib.load(filepath)

        # Filtragem booleana idempotente das classes ocultadas
        mask = ~np.isin(y_train, hidden_classes)
        X_train_masked, y_train_masked = X_train[mask], y_train[mask]

        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=60,
            early_stopping=True,
            random_state=self.random_state
        )
        model.fit(X_train_masked, y_train_masked)
        joblib.dump(model, filepath)
        return model

    def evaluate_ood_generalization(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        hidden_classes: List[int] = [4, 7]
    ) -> np.ndarray:
        """
        Avalia o comportamento de 'falsa certeza' em classes não vistas no treino (Desafio B).
        """
        ood_mask = np.isin(y_test, hidden_classes)
        X_test_ood, y_test_ood = X_test[ood_mask], y_test[ood_mask]

        y_pred_ood = model.predict(X_test_ood)
        cm = confusion_matrix(y_test_ood, y_pred_ood, labels=list(range(10)))

        plt.figure(figsize=(8, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=[str(i) for i in range(10)],
            yticklabels=[str(i) for i in range(10)]
        )
        plt.title(f"Matriz OOD - Classes Ocultadas {hidden_classes}", fontsize=11, fontweight="bold")
        plt.xlabel("Rótulo Preditivo (Atribuído pelo Modelo)", fontsize=9)
        plt.ylabel("Rótulo Verdadeiro OOD", fontsize=9)
        plt.tight_layout()
        plt.show()

        return cm

    def preprocess_custom_image(self, image_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        DESAFIO C - Pipeline de Normalização Dinâmica e Padronização de Fundo:
        1. Leitura do arquivo e conversão para escala de cinza.
        2. Avaliação da luminância média: se fundo for claro (>127), inverte dinamicamente.
        3. Limiarização Otsu para isolamento do traço sem ruídos de compressão.
        4. Enquadramento proporcional com margem de respiro (Square Canvas Padding).
        5. Redimensionamento 28x28 e normalização no intervalo contínuo [0.0, 1.0].
        """
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Imagem não encontrada no caminho especificado: {image_path}")

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Padronização Dinâmica: Se o fundo for claro (média > 127), inverte para fundo escuro + traço claro
        if np.mean(gray) > 127:
            gray_processed = 255 - blurred
        else:
            gray_processed = blurred.copy()

        # Limiarização Binarizada de Otsu
        _, thresh = cv2.threshold(gray_processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Filtra contornos irrelevantes descartando ruídos de fundo inferiores a 30 pixels
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 30]

        if valid_contours:
            c = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            digit_roi = thresh[y:y+h, x:x+w]

            # Container quadrado proporcional (Square Canvas Padding)
            max_dim = max(w, h)
            margin = int(max_dim * 0.25)
            square_dim = max_dim + (2 * margin)

            square_canvas = np.zeros((square_dim, square_dim), dtype=np.uint8)
            start_x = margin + (max_dim - w) // 2
            start_y = margin + (max_dim - h) // 2
            square_canvas[start_y:start_y+h, start_x:start_x+w] = digit_roi

            img_28x28 = cv2.resize(square_canvas, (28, 28), interpolation=cv2.INTER_AREA)
        else:
            img_28x28 = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)

        # Vetorização e normalização exigida pelos modelos de Machine Learning [0.0, 1.0]
        normalized_flat = (img_28x28.astype(np.float32) / 255.0).reshape(1, 784)
        return normalized_flat, img_28x28

    def plot_multiple_custom_inferences(
        self,
        model: Any,
        images_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """
        Plota a grade comparativa 4x2 com a imagem de entrada e as probabilidades Softmax.
        """
        num_images = len(images_dict)
        fig, axes = plt.subplots(num_images, 2, figsize=(11, 3.8 * num_images))

        if num_images == 1:
            axes = np.expand_dims(axes, axis=0)

        for idx, (bg_label, (X_custom, img_28x28)) in enumerate(images_dict.items()):
            probs = model.predict_proba(X_custom)[0]
            pred_digit = np.argmax(probs)
            confidence = probs[pred_digit] * 100

            ax_img = axes[idx, 0]
            ax_bar = axes[idx, 1]

            ax_img.imshow(img_28x28, cmap="gray")
            ax_img.set_title(
                f"Fundo: {bg_label}\nPredição: Dígito {pred_digit} ({confidence:.1f}%)",
                fontweight="bold", fontsize=10
            )
            ax_img.axis("off")

            bars = ax_bar.bar(range(10), probs * 100, color="skyblue")
            bars[pred_digit].set_color("crimson")
            ax_bar.set_xticks(range(10))
            ax_bar.set_xlabel("Classe (Dígito)", fontsize=9)
            ax_bar.set_ylabel("Probabilidade (%)", fontsize=9)
            ax_bar.set_ylim(0, 105)

            for bar in bars:
                yval = bar.get_height()
                if yval > 3.0:
                    ax_bar.text(
                        bar.get_x() + bar.get_width()/2.0,
                        yval + 1.5,
                        f"{yval:.1f}%",
                        ha="center", va="bottom", fontsize=8
                    )

        plt.suptitle("Comparativo de Inferência Externa sob Padronização Dinâmica de Fundo", fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.show()
