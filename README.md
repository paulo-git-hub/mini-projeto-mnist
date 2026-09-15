# 📌 MINIPROJETO AVALIATIVO — MÓDULO 2
## Sistema de Classificação MNIST, Benchmarking de Modelos e Testes de Robustez Out-Of-Distribution (OOD)

### 🎯 Objetivos do Projeto
Este projeto desenvolve uma solução completa e ponta a ponta de Machine Learning para o reconhecimento de dígitos manuscritos (dataset `mnist_784`). Todo o ambiente foi arquitetado para execução em nuvem através do **Google Colab**, com persistência de dados no **Google Drive** (`/content/drive/MyDrive/mini_projeto_mnist/`). A solução combina engenharia de software rigorosa, Arquitetura Orientada a Objetos (POO), execução idempotente e controle de versão automatizado via Git Flow integrado ao GitHub.

---

### 🔬 Escopo e Pilares Técnicos

* **Infraestrutura Colab & Drive:** Operação nativa no Google Colab com montagem do Google Drive, garantindo a persistência absoluta de scripts (`src/`), modelos treinados (`models/`) e datasets (`data/`) para evitar reprocessamentos (Idempotência).
* **Engenharia de Dados e EDA:** Ingestão idempotente do dataset MNIST com suporte a cache local, verificação de integridade, divisão estratificada de classes (`stratify=y`) e normalização contínua de pixels.
* **Benchmarking Multimodelo:** Implementação, otimização e serialização das arquiteturas de classificadores exigidos: **KNN ($k=3$)**, **Random Forest (100 árvores)** e **Perceptron Multicamadas (MLP Classifier)**.
* **Avaliação Métrica Sistemática:** Comparação rigorosa de desempenho via Acurácia, Precisão, Recall, F1-Score Macro e Matrizes de Confusão $10 \times 10$ com mapas de calor.
* **Estresse e Robustez OOD (Desafios A, B e C):** Investigação do fenômeno de *overconfidence* através do mascaramento de classes (ausência dos dígitos 4 e 7 no treino) e desenvolvimento de pipeline de Visão Computacional (OpenCV) para ingestão externa (Inversão Dinâmica de Polaridade, Binarização de Otsu e *Square Canvas Padding*).
* **Deploy e Interface em Tempo Real:** Disponibilização do modelo preditivo em uma aplicação web interativa desenvolvida em **Gradio**, suportando upload de arquivos externos e desenho em tela viva (*Sketchpad*).

---

### 🛠️ Arquitetura e Stack Tecnológica

| Camada da Solução | Tecnologias e Paradigmas |
| :--- | :--- |
| **Infraestrutura em Nuvem** | Google Colab, Google Drive API (`google.colab.drive`) |
| **Linguagem & Paradigma** | Python 3.10+ (POO, Type Hints, Idempotência Absoluta) |
| **Machine Learning & Core** | Scikit-learn, NumPy, Pandas, Joblib |
| **Visão Computacional & EDA** | OpenCV (`cv2`), Pillow, Matplotlib, Seaborn |
| **Frontend Interativo** | Gradio (UI responsiva com Blocks e reset de Canvas) |
| **Versionamento & CI/CD** | Git, GitHub (Git Flow automatizado via módulo `GitManager`) |

---

### 📊 Resumo Executivo de Desempenho (Benchmark Test)

A avaliação preditiva no conjunto de teste (14.000 amostras) comprovou a superioridade da rede neural MLP:

| Modelo | Acurácia Global | Precisão Macro | Recall Macro | F1-Score Macro |
| :--- | :---: | :---: | :---: | :---: |
| **MLP (Perceptron Multicamadas)** | **97.76%** | **97.76%** | **97.74%** | **97.75%** |
| **KNN (K-Nearest Neighbors)** | 97.12% | 97.16% | 97.08% | 97.11% |
| **Random Forest (100 árvores)** | 96.63% | 96.62% | 96.60% | 96.61% |

---

### 🔬 Análise de Robustez e Inferência Externa (Desafio C)

O modelo líder (MLP) foi submetido ao pipeline OpenCV com **Padronização Dinâmica de Fundo**. A avaliação de imagens externas atingiu **75% de sucesso**:

* 📝 **Resiliência a Texturas Ruidosas (Dígito 2 Textura jpg):** Mesmo com ruídos e artefatos de fundo remanescentes após o Otsu, o pipeline preservou os traços essenciais do dígito 2 (curva superior e base plana). A MLP classificou a imagem com **100.0% de confiança**, demonstrando alta tolerância quando a topologia do caractere é preservada.
* 📝 **Inversão Dinâmica de Fundo (Dígito 5 Branco png):** Amostra manuscrita/digitalizada em fundo claro. O algoritmo detectou a alta luminância, inverteu a polaridade para fundo escuro com traço claro e enquadrou a matriz $28 \times 28$, obtendo **99.6% de acurácia**.
* 🖥️ **Amostra Sintética Limpa (Dígito 0 Preto jpg):** Imagem sintética ideal em fundo preto nativo, atingindo **99.9% de precisão**.
* ⚠️ **Sensibilidade a Ruídos Topológicos (Dígito 7 Textura jpg):** Única falha observada (predição incorreta como Dígito 8 com 52.3%). A alta frequência do fundo ruidoso gerou blocos de pixels que se conectaram ao traço do 7. A MLP, atuando sobre vetores de 784 dimensões sem filtros convolucionais, interpretou esses artefatos como alças fechadas.

---

### 🚀 Conclusão e Próximos Passos

A solução atendeu a todos os requisitos arquiteturais e de código. O pipeline de ingestão provou ser robusto para imagens do mundo real (mesmo com fundo claro ou texturizado). Para trabalhos futuros, recomenda-se a transição para uma **Rede Neural Convolucional (CNN)** para adicionar invariância espacial absoluta contra degradações topológicas severas.
