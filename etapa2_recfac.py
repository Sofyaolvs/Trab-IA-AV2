"""
Etapa 2 - Classificacao Multiclasse: Reconhecimento Facial (20 classes)
Modelos: Perceptron Simples, ADALINE, MLP, RBF
Bibliotecas permitidas: numpy, matplotlib, seaborn, opencv
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import time

np.random.seed(42)

# ==============================================================================
# 1. CARREGAMENTO E PREPROCESSAMENTO DAS IMAGENS
# ==============================================================================

RECFAC_PATH = os.path.join(os.path.dirname(__file__), "RecFac")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "resultados_etapa2")
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = 50  # Redimensionar para 50x50
p = IMG_SIZE * IMG_SIZE  # 2500 atributos

classes = sorted([d for d in os.listdir(RECFAC_PATH)
                  if os.path.isdir(os.path.join(RECFAC_PATH, d))])
C = len(classes)
print(f"Classes encontradas ({C}): {classes}")

images = []
labels = []
for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(RECFAC_PATH, class_name)
    img_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.png')])
    for img_file in img_files:
        img_path = os.path.join(class_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_vector = img_resized.flatten().astype(np.float64)
        images.append(img_vector)
        labels.append(class_idx)

X_raw = np.array(images)   # (640, 2500)
y_indices = np.array(labels)  # (640,) indices 0..19
N_total = X_raw.shape[0]

print(f"Total de imagens: {N_total}, Dimensao: {p}")
print(f"Imagens por classe: {N_total // C}")

# Visualizar amostras
fig, axes = plt.subplots(2, 10, figsize=(15, 4))
for i, class_name in enumerate(classes):
    row, col = i // 10, i % 10
    idx = np.where(y_indices == i)[0][0]
    axes[row, col].imshow(X_raw[idx].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    axes[row, col].set_title(class_name, fontsize=7)
    axes[row, col].axis('off')
plt.suptitle("Amostra de cada classe (50x50)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "01_amostras_faces.png"), dpi=150, bbox_inches='tight')
plt.close()


# ==============================================================================
# 2. ONE-HOT ENCODING (+1/-1) e NORMALIZACAO
# ==============================================================================

def one_hot_encode(y_idx, num_classes):
    N = len(y_idx)
    Y = -np.ones((N, num_classes))
    for i in range(N):
        Y[i, y_idx[i]] = 1.0
    return Y

Y_onehot = one_hot_encode(y_indices, C)  # (640, 20)
print(f"Y one-hot shape: {Y_onehot.shape}")

def normalize_minmax(X, x_min=None, x_max=None):
    if x_min is None:
        x_min = X.min(axis=0)
    if x_max is None:
        x_max = X.max(axis=0)
    denom = x_max - x_min
    denom[denom == 0] = 1
    return (X - x_min) / denom, x_min, x_max

def add_bias(X):
    return np.hstack([-np.ones((X.shape[0], 1)), X])


# ==============================================================================
# 3. FUNCOES AUXILIARES
# ==============================================================================

def confusion_matrix_manual(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, pr in zip(y_true, y_pred):
        cm[int(t), int(pr)] += 1
    return cm

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def plot_confusion_matrix_multi(cm, class_names, title, save_path=None):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(title)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=7)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_learning_curve(errors, title, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(errors, linewidth=0.8)
    plt.xlabel("Epoca")
    plt.ylabel("Erro (EQM)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================
# 4. MODELOS DE RNA (Multiclasse, otimizados para alta dimensao)
# ==============================================================================

# ---------- PERCEPTRON SIMPLES (Multiclasse) ----------
class PerceptronSimples:
    def __init__(self, lr=0.01, max_epochs=100):
        self.lr = lr
        self.max_epochs = max_epochs
        self.weights = None
        self.errors_per_epoch = []

    def fit(self, X, Y):
        """X: (N, p+1) com bias, Y: (N, C) one-hot +1/-1."""
        N, d = X.shape
        C = Y.shape[1]
        self.weights = np.random.uniform(-0.5, 0.5, (d, C))
        self.errors_per_epoch = []

        for epoch in range(self.max_epochs):
            U = X @ self.weights  # (N, C)
            Y_pred = np.where(U >= 0, 1.0, -1.0)
            diff = Y - Y_pred  # (N, C)
            wrong = np.any(diff != 0, axis=1)
            error_count = int(np.sum(wrong))
            if error_count > 0:
                # Batch update from wrong samples
                self.weights += (self.lr / error_count) * (X[wrong].T @ diff[wrong])
            self.errors_per_epoch.append(error_count / N)
            if error_count == 0:
                break

    def predict(self, X):
        return np.argmax(X @ self.weights, axis=1)


# ---------- ADALINE (Multiclasse - Batch) ----------
class ADALINE:
    def __init__(self, lr=0.001, max_epochs=100, tol=1e-6):
        self.lr = lr
        self.max_epochs = max_epochs
        self.tol = tol
        self.weights = None
        self.errors_per_epoch = []

    def fit(self, X, Y):
        N, d = X.shape
        C = Y.shape[1]
        self.weights = np.random.uniform(-0.5, 0.5, (d, C))
        self.errors_per_epoch = []

        for epoch in range(self.max_epochs):
            U = X @ self.weights  # (N, C)
            error = Y - U  # (N, C)
            eqm = np.sum(error ** 2) / (2 * N)
            self.errors_per_epoch.append(eqm)
            # Batch gradient
            grad = -(X.T @ error) / N  # (d, C)
            self.weights -= self.lr * grad
            if epoch > 0 and abs(self.errors_per_epoch[-1] - self.errors_per_epoch[-2]) < self.tol:
                break

    def predict(self, X):
        return np.argmax(X @ self.weights, axis=1)


# ---------- MLP (Multiclasse - Mini-batch otimizado) ----------
class MLP:
    def __init__(self, hidden_layers=(100,), lr=0.01, max_epochs=200, tol=1e-6,
                 activation='tanh', batch_size=32):
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.max_epochs = max_epochs
        self.tol = tol
        self.activation = activation
        self.batch_size = batch_size
        self.weights = []
        self.biases = []
        self.errors_per_epoch = []

    def _activate(self, z):
        if self.activation == 'tanh':
            return np.tanh(z)
        else:
            return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _activate_deriv(self, a):
        if self.activation == 'tanh':
            return 1.0 - a ** 2
        else:
            return a * (1.0 - a)

    def fit(self, X, Y):
        """X: (N, p) SEM bias, Y: (N, C)."""
        N, pin = X.shape
        C = Y.shape[1]

        layers = [pin] + list(self.hidden_layers) + [C]
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            limit = np.sqrt(6.0 / (layers[i] + layers[i + 1]))
            self.weights.append(np.random.uniform(-limit, limit, (layers[i], layers[i + 1])))
            self.biases.append(np.zeros((1, layers[i + 1])))

        self.errors_per_epoch = []

        for epoch in range(self.max_epochs):
            idx = np.random.permutation(N)
            epoch_error = 0.0

            # Mini-batch training
            for start in range(0, N, self.batch_size):
                batch_idx = idx[start:start + self.batch_size]
                xb = X[batch_idx]  # (B, p)
                yb = Y[batch_idx]  # (B, C)
                B = xb.shape[0]

                # Forward
                activations = [xb]
                for l in range(len(self.weights)):
                    z = activations[-1] @ self.weights[l] + self.biases[l]
                    a = self._activate(z)
                    activations.append(a)

                output = activations[-1]
                error = yb - output
                epoch_error += np.sum(error ** 2)

                # Backward
                delta = error * self._activate_deriv(output)
                for l in range(len(self.weights) - 1, -1, -1):
                    self.weights[l] += (self.lr / B) * (activations[l].T @ delta)
                    self.biases[l] += (self.lr / B) * np.sum(delta, axis=0, keepdims=True)
                    if l > 0:
                        delta = (delta @ self.weights[l].T) * self._activate_deriv(activations[l])

            eqm = epoch_error / (2 * N)
            self.errors_per_epoch.append(eqm)
            if epoch > 0 and abs(self.errors_per_epoch[-1] - self.errors_per_epoch[-2]) < self.tol:
                break

    def predict(self, X):
        a = X
        for l in range(len(self.weights)):
            z = a @ self.weights[l] + self.biases[l]
            a = self._activate(z)
        return np.argmax(a, axis=1)


# ---------- RBF (Multiclasse - Otimizado) ----------
class RBF:
    def __init__(self, n_centers=50, lr=0.001, max_epochs=100, tol=1e-6):
        self.n_centers = n_centers
        self.lr = lr
        self.max_epochs = max_epochs
        self.tol = tol
        self.centers = None
        self.sigmas = None
        self.weights = None
        self.errors_per_epoch = []

    def _kmeans(self, X, k, max_iter=50):
        N = X.shape[0]
        idx = np.random.choice(N, k, replace=False)
        centers = X[idx].copy()
        for it in range(max_iter):
            # Distancias vetorizadas em blocos para evitar memoria excessiva
            labels = np.zeros(N, dtype=int)
            block = 100
            for s in range(0, N, block):
                e = min(s + block, N)
                diff = X[s:e, None, :] - centers[None, :, :]  # (block, k, p)
                dists = np.sum(diff ** 2, axis=2)  # (block, k)
                labels[s:e] = np.argmin(dists, axis=1)
            new_centers = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j]
                                    for j in range(k)])
            if np.allclose(centers, new_centers, atol=1e-4):
                break
            centers = new_centers
        return centers

    def _compute_H(self, X):
        """Calcula matriz de ativacao H de forma vetorizada."""
        N = X.shape[0]
        H = np.zeros((N, self.n_centers))
        for j in range(self.n_centers):
            diff = X - self.centers[j]  # (N, p)
            dist_sq = np.sum(diff ** 2, axis=1)  # (N,)
            H[:, j] = np.exp(-dist_sq / (2 * self.sigmas[j] ** 2 + 1e-10))
        return H

    def fit(self, X, Y):
        """X: (N, p) SEM bias, Y: (N, C)."""
        N = X.shape[0]

        # K-Means
        self.centers = self._kmeans(X, self.n_centers)

        # Sigma: media das distancias aos 2 centros mais proximos
        dists_c = np.zeros((self.n_centers, self.n_centers))
        for i in range(self.n_centers):
            for j in range(self.n_centers):
                dists_c[i, j] = np.sqrt(np.sum((self.centers[i] - self.centers[j]) ** 2))
        np.fill_diagonal(dists_c, np.inf)
        self.sigmas = np.zeros(self.n_centers)
        for j in range(self.n_centers):
            k_nn = min(2, self.n_centers - 1)
            self.sigmas[j] = np.mean(np.sort(dists_c[j])[:k_nn])
        self.sigmas[self.sigmas == 0] = 1.0

        # Camada oculta
        H = self._compute_H(X)
        H_bias = np.hstack([H, np.ones((N, 1))])

        # Pseudo-inversa para pesos iniciais
        self.weights = np.linalg.pinv(H_bias) @ Y

        y_pred = H_bias @ self.weights
        initial_error = np.sum((Y - y_pred) ** 2) / (2 * N)
        self.errors_per_epoch = [initial_error]

        # Refinamento com gradiente (batch)
        for epoch in range(self.max_epochs):
            output = H_bias @ self.weights  # (N, C)
            error = Y - output
            eqm = np.sum(error ** 2) / (2 * N)
            self.errors_per_epoch.append(eqm)

            # Batch gradient update
            grad = -(H_bias.T @ error) / N  # (n_centers+1, C)
            self.weights -= self.lr * grad

            if abs(self.errors_per_epoch[-1] - self.errors_per_epoch[-2]) < self.tol:
                break

    def predict(self, X):
        H = self._compute_H(X)
        H_bias = np.hstack([H, np.ones((X.shape[0], 1))])
        output = H_bias @ self.weights
        return np.argmax(output, axis=1)


# ==============================================================================
# 5. VALIDACAO MONTE CARLO
# ==============================================================================

print("\n=== VALIDACAO MONTE CARLO ===")

model_configs = {
    'Perceptron Simples': lambda: PerceptronSimples(lr=0.01, max_epochs=50),
    'ADALINE':            lambda: ADALINE(lr=0.001, max_epochs=50, tol=1e-6),
    'MLP':                lambda: MLP(hidden_layers=(100, 50), lr=0.01, max_epochs=100,
                                      activation='tanh', batch_size=64),
    'RBF':                lambda: RBF(n_centers=40, lr=0.01, max_epochs=50),
}

R_total = 100
results = {name: [] for name in model_configs}
detail_data = {name: {'best_acc': -1, 'worst_acc': 2,
                       'best_round': None, 'worst_round': None,
                       'best_cm': None, 'worst_cm': None,
                       'best_errors': None, 'worst_errors': None}
               for name in model_configs}

for r in range(R_total):
    t0 = time.time()
    print(f"\n--- Rodada {r + 1}/{R_total} ---")

    idx_perm = np.random.permutation(N_total)
    n_tr = int(0.8 * N_total)
    tr_idx, te_idx = idx_perm[:n_tr], idx_perm[n_tr:]

    X_tr_raw, X_te_raw = X_raw[tr_idx], X_raw[te_idx]
    y_tr_idx, y_te_idx = y_indices[tr_idx], y_indices[te_idx]
    Y_tr_oh, Y_te_oh = Y_onehot[tr_idx], Y_onehot[te_idx]

    X_tr_norm, xmin, xmax = normalize_minmax(X_tr_raw)
    X_te_norm, _, _ = normalize_minmax(X_te_raw, xmin, xmax)

    X_tr_bias = add_bias(X_tr_norm)
    X_te_bias = add_bias(X_te_norm)

    for model_name, model_fn in model_configs.items():
        tm = time.time()
        model = model_fn()

        if model_name in ['Perceptron Simples', 'ADALINE']:
            model.fit(X_tr_bias, Y_tr_oh)
            y_pred_idx = model.predict(X_te_bias)
        else:
            model.fit(X_tr_norm, Y_tr_oh)
            y_pred_idx = model.predict(X_te_norm)

        acc = accuracy_score(y_te_idx, y_pred_idx)
        results[model_name].append(acc)
        print(f"  {model_name}: acc={acc:.4f} ({time.time()-tm:.1f}s)")

        dd = detail_data[model_name]
        if acc > dd['best_acc']:
            dd['best_acc'] = acc
            dd['best_round'] = r
            dd['best_cm'] = confusion_matrix_manual(y_te_idx, y_pred_idx, C)
            dd['best_errors'] = model.errors_per_epoch.copy()
        if acc < dd['worst_acc']:
            dd['worst_acc'] = acc
            dd['worst_round'] = r
            dd['worst_cm'] = confusion_matrix_manual(y_te_idx, y_pred_idx, C)
            dd['worst_errors'] = model.errors_per_epoch.copy()

    print(f"  Rodada total: {time.time()-t0:.1f}s")


# ==============================================================================
# 6. MATRIZES DE CONFUSAO E CURVAS (MELHOR E PIOR)
# ==============================================================================

print("\n=== MELHOR E PIOR RODADA POR MODELO ===")
for model_name in model_configs:
    dd = detail_data[model_name]
    print(f"\n{model_name}:")
    print(f"  Melhor acuracia: {dd['best_acc']:.4f} (rodada {dd['best_round']})")
    print(f"  Pior acuracia:   {dd['worst_acc']:.4f} (rodada {dd['worst_round']})")

    safe_name = model_name.replace(' ', '_')

    plot_confusion_matrix_multi(dd['best_cm'], classes,
                                f"{model_name} - Melhor Acc ({dd['best_acc']:.4f})",
                                os.path.join(RESULTS_DIR, f"02_{safe_name}_melhor_cm.png"))
    plot_confusion_matrix_multi(dd['worst_cm'], classes,
                                f"{model_name} - Pior Acc ({dd['worst_acc']:.4f})",
                                os.path.join(RESULTS_DIR, f"02_{safe_name}_pior_cm.png"))

    if dd['best_errors']:
        plot_learning_curve(dd['best_errors'],
                            f"{model_name} - Curva (Melhor Rodada)",
                            os.path.join(RESULTS_DIR, f"02_{safe_name}_melhor_curva.png"))
    if dd['worst_errors']:
        plot_learning_curve(dd['worst_errors'],
                            f"{model_name} - Curva (Pior Rodada)",
                            os.path.join(RESULTS_DIR, f"02_{safe_name}_pior_curva.png"))

    # Analise por classe
    print(f"\n  Analise de categorias ({model_name} - Melhor Rodada):")
    diag = np.diag(dd['best_cm'])
    row_sums = dd['best_cm'].sum(axis=1)
    per_class_acc = np.where(row_sums > 0, diag / row_sums, 0)
    best_cls = np.argsort(per_class_acc)[::-1][:5]
    worst_cls = np.argsort(per_class_acc)[:5]
    print(f"    Melhores: {[(classes[i], f'{per_class_acc[i]:.2f}') for i in best_cls]}")
    print(f"    Piores:   {[(classes[i], f'{per_class_acc[i]:.2f}') for i in worst_cls]}")


# ==============================================================================
# 7. TABELA FINAL - ESTATISTICAS
# ==============================================================================

print("\n=== TABELA DE RESULTADOS (100 RODADAS) - ACURACIA ===")
print(f"{'Modelo':<35} {'Media':>10} {'Desvio':>10} {'Maior':>10} {'Menor':>10}")
print("-" * 77)
for model_name in model_configs:
    vals = np.array(results[model_name])
    print(f"{model_name:<35} {vals.mean():>10.4f} {vals.std():>10.4f} "
          f"{vals.max():>10.4f} {vals.min():>10.4f}")

# Boxplot
plt.figure(figsize=(10, 6))
data_box = [np.array(results[name]) for name in model_configs]
bp = plt.boxplot(data_box, tick_labels=[n.replace(' ', '\n') for n in model_configs], patch_artist=True)
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
plt.title("Acuracia dos Modelos - 100 Rodadas (Reconhecimento Facial)")
plt.ylabel("Acuracia")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(RESULTS_DIR, "03_boxplot_acuracia.png"), dpi=150, bbox_inches='tight')
plt.close()

# Violin plot
plt.figure(figsize=(10, 6))
parts = plt.violinplot(data_box, showmeans=True, showmedians=True)
plt.xticks(range(1, len(model_configs) + 1),
           [n.replace(' ', '\n') for n in model_configs])
plt.title("Violin Plot - Acuracia (Reconhecimento Facial)")
plt.ylabel("Acuracia")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(RESULTS_DIR, "03_violinplot_acuracia.png"), dpi=150, bbox_inches='tight')
plt.close()

print("\nTodos os resultados salvos em:", RESULTS_DIR)
print("Etapa 2 concluida com sucesso!")
