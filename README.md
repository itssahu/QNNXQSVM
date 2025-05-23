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

![image](https://github.com/user-attachments/assets/3b1fe19e-ecac-42c6-b8ba-9d6678025f42)


---

### 🔹 30% Noise

- QSVM misclassifies both Versicolor and Virginica.
- QNN: Slight confusion (1 misclassification); high accuracy.

![image](https://github.com/user-attachments/assets/39c2799d-5b49-49b1-9d35-d8093e2611d5)


---

### 🔹 50% Noise

- QSVM accuracy nearly collapses to ~40%, large confusion between classes.
- QNN maintains over 90% accuracy, PCA still shows separation.

![image](https://github.com/user-attachments/assets/9d4200cd-d685-48f6-aa84-46a6cc8ffadd)


---

### 🔹 70% Noise

- QSVM shows noisy PCA and indistinct clusters.
- QNN confusion matrix shows 2 errors total.


![image](https://github.com/user-attachments/assets/0cef5256-ae7b-49f7-a7c3-9cf0471d2c16)

---

### 🔹 90% Noise

- QSVM: Kernel matrix highly noisy; PCA fully overlapping.
- QNN: Accuracy drops but remains significantly better.

![image](https://github.com/user-attachments/assets/13e9ab76-37ec-4e88-a38a-08bf20a37dfd)


---

### 🔹 100% Noise

- QSVM performance close to random guessing.
- QNN still distinguishes structure using entangling layer.

![image](https://github.com/user-attachments/assets/d0cd9d53-5248-48fc-970d-a449a1daf573)


---

## 5. 📈 Accuracy vs Noise Level

QNN maintains **>80% accuracy** up to 60% noise, while QSVM drops to ~30% quickly.

![image](https://github.com/user-attachments/assets/f33c89e8-2f33-4531-92c9-17cc434f77b2)


---

## 6. ⏱️ Execution Time vs Noise Level

- QSVM executes in ~4–5s across all noise levels.
- QNN execution varies (35–100s), due to iterative optimization.

![image](https://github.com/user-attachments/assets/ca45d12e-815d-480c-8857-a4a95be2d5e8)


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
│   └── train_qnn_qsvm.ipynb
├── requirements.txt
└── README.md
```

---

## 10. 📄 License

This project is licensed under the MIT License.

