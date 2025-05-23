# QNN vs QSVM: Robustness Benchmarking on Noisy Iris Dataset

This repository presents a comprehensive benchmarking study comparing the robustness of **Quantum Neural Networks (QNN)** and **Quantum Support Vector Machines (QSVM)** under increasing input noise, using the classic Iris classification dataset.

---

## 1. 🎯 Objective

- Analyze and compare QNN and QSVM performance on **multiclass classification**.
- Systematically inject **Gaussian noise** at levels from **10% to 100%**.
- Evaluate models on:
  - Accuracy
  - Confusion matrix
  - PCA separation
  - Quantum kernel behavior (QSVM)
  - Quantum weights (QNN)
  - Execution time

---

## 2. 📊 Dataset

- **Name**: Iris dataset (`sklearn.datasets.load_iris`)
- **Classes**: 3 (Setosa, Versicolor, Virginica)
- **Features**: 4 continuous (sepal & petal dimensions)
- **Size**: 150 samples (50 per class)

---

## 3. 🧪 Experimental Design

### Noise Injection
Each experiment perturbs features with Gaussian noise:
```python
X_noisy = X_clean + np.random.normal(0, noise_level × X.std(), X.shape)
```
- Noise Levels Tested: **10%, 20%, ..., 100%**

---

### 3.1 QSVM Setup

| Component        | Value |
|------------------|-------|
| Feature Map      | `ZZFeatureMap(d=4, reps=2, entanglement='full')` |
| Backend          | `BasicAer.statevector_simulator` |
| Kernel Function  | `QuantumKernel.evaluate()` |
| Classifier       | `sklearn.svm.SVC` |

---

### 3.2 QNN Setup

| Component           | Architecture                    |
|---------------------|----------------------------------|
| Qubits              | 4 wires                          |
| Embedding           | `qml.AngleEmbedding`             |
| Circuit             | `qml.StronglyEntanglingLayers`   |
| Output Layer        | `nn.Linear(4, 3)`                |
| Optimizer           | `Adam (lr=0.01)`                 |
| Regularization      | Early stopping (patience = 10)  |

---

## 4. 📌 Results by Noise Level

### 🔹 10% Noise

| QSVM                             | QNN                              |
|----------------------------------|----------------------------------|
| Confusion: Some class mix        | Confusion: Perfect classification |
| PCA: Overlap in clusters         | PCA: Well-separated clusters     |
| Kernel: Diagonal but noisy       | Weights: Balanced and smooth     |

![noise10](results/10_percent.png)

---

### 🔹 30% Noise

- QSVM misclassifies both Versicolor and Virginica.
- QNN: Slight confusion (1 misclassification); high accuracy.

![noise30](results/30_percent.png)

---

### 🔹 50% Noise

- QSVM accuracy nearly collapses to ~40%, large confusion between classes.
- QNN maintains over 90% accuracy, PCA still shows separation.

![noise50](results/50_percent.png)

---

### 🔹 70% Noise

- QSVM shows noisy PCA and indistinct clusters.
- QNN confusion matrix shows 2 errors total.

![noise70](results/70_percent.png)

---

### 🔹 90% Noise

- QSVM: Kernel matrix highly noisy; PCA fully overlapping.
- QNN: Accuracy drops but remains significantly better.

![noise90](results/90_percent.png)

---

### 🔹 100% Noise

- QSVM performance close to random guessing.
- QNN still distinguishes structure using entangling layer.

![noise100](results/100_percent.png)

---

## 5. 📈 Accuracy vs Noise Level

QNN maintains **>80% accuracy** up to 60% noise, while QSVM drops to ~30% quickly.

![accuracy](results/accuracy_vs_noise.png)

---

## 6. ⏱️ Execution Time vs Noise Level

- QSVM executes in ~4–5s across all noise levels.
- QNN execution varies (35–100s), due to iterative optimization.

![time](results/time_vs_noise.png)

---

## 7. 💡 Key Insights

| Aspect             | QSVM                             | QNN                               |
|--------------------|----------------------------------|------------------------------------|
| Accuracy (low noise) | Moderate                        | Perfect (100%)                     |
| Accuracy (high noise) | Poor (random-like)             | Stable (60–80%)                    |
| Kernel Matrix       | Degrades quickly                | —                                  |
| Quantum Weights     | —                               | Adaptively tuned                   |
| PCA Separation      | Collapses                       | Maintains structure                |
| Training Time       | Fast (~5s)                      | Slower (~40–100s)                  |

---

## 8. 📦 Running This Project

```bash
git clone https://github.com/YOUR_USERNAME/qnn-vs-qsvm-robustness.git
cd qnn-vs-qsvm-robustness
pip install -r requirements.txt
python src/train_qnn_qsvm.py
```

---

## 9. 📂 Repository Structure

```
.
├── src/
│   └── train_qnn_qsvm.py
├── results/
│   ├── 10_percent.png
│   ├── 30_percent.png
│   ├── ...
│   ├── accuracy_vs_noise.png
│   └── time_vs_noise.png
├── requirements.txt
└── README.md
```

---

## 10. 📄 License

This project is licensed under the MIT License.

