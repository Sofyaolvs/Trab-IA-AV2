"""
Etapa 1 - Classificacao Binaria: Spiral Dataset
Modelos: Perceptron Simples, ADALINE, MLP
Bibliotecas permitidas: numpy, matplotlib, seaborn
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

np.random.seed(42)

# ==============================================================================
# 1. CARREGAMENTO E ORGANIZACAO DOS DADOS
# ==============================================================================

DATA_PATH = os.path.join(os.path.dirname(__file__), "spiral_d (1).csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "resultados_etapa1")
os.makedirs(RESULTS_DIR, exist_ok=True)

data = np.loadtxt(DATA_PATH, delimiter=",")
X_raw = data[:, :2]   # (N, 2)
y_raw = data[:, 2]    # (N,) valores: -1 e 1

N_total = X_raw.shape[0]
p = X_raw.shape[1]

print(f"Dados carregados: {N_total} amostras, {p} atributos")
print(f"Classes: {np.unique(y_raw)}, Contagem: { {v: int((y_raw==v).sum()) for v in np.unique(y_raw)} }")


# ==============================================================================
# 2. NORMALIZACAO e BIAS
# ==============================================================================

def normalize_minmax(X, x_min=None, x_max=None):
    if x_min is None:
        x_min = X.min(axis=0)
    if x_max is None:
        x_max = X.max(axis=0)
    denom = x_max - x_min
    denom[denom == 0] = 1
    return (X - x_min) / denom, x_min, x_max

def add_bias(X):
    """Adiciona coluna de -1 (bias) como primeira coluna."""
    return np.hstack([-np.ones((X.shape[0], 1)), X])


# ==============================================================================
# 3. VISUALIZACAO INICIAL
# ==============================================================================

plt.figure(figsize=(8, 6))
mask_pos = y_raw == 1
mask_neg = y_raw == -1
plt.scatter(X_raw[mask_pos, 0], X_raw[mask_pos, 1], c='blue', s=10, alpha=0.6, label='Classe +1')
plt.scatter(X_raw[mask_neg, 0], X_raw[mask_neg, 1], c='red', s=10, alpha=0.6, label='Classe -1')
plt.xlabel("x1"); plt.ylabel("x2")
plt.title("Espalhamento dos Dados Spiral")
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "01_scatter_spiral.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Grafico de espalhamento salvo.")


# ==============================================================================
# 4. FUNCOES AUXILIARES: METRICAS
# ==============================================================================

def confusion_matrix_manual(y_true, y_pred, classes):
    C = len(classes)
    cm = np.zeros((C, C), dtype=int)
    c2i = {c: i for i, c in enumerate(classes)}
    for t, pr in zip(y_true, y_pred):
        cm[c2i[t], c2i[pr]] += 1
    return cm

def binary_metrics(y_true, y_pred):
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == -1) & (y_pred == -1)))
    FP = int(np.sum((y_true == -1) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == -1)))
    acc = (TP + TN) / max(TP + TN + FP + FN, 1)
    sens = TP / max(TP + FN, 1)
    spec = TN / max(TN + FP, 1)
    prec = TP / max(TP + FP, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-10)
    return {'acuracia': acc, 'sensibilidade': sens, 'especificidade': spec,
            'precisao': prec, 'f1_score': f1}

def plot_cm(cm, cls_labels, title, path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=cls_labels, yticklabels=cls_labels)
    plt.xlabel("Predito"); plt.ylabel("Real"); plt.title(title)
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()

def plot_curve(errors, title, path):
    plt.figure(figsize=(8, 5))
    plt.plot(errors, linewidth=0.8)
    plt.xlabel("Epoca"); plt.ylabel("Erro"); plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()


# ==============================================================================
# 5. MODELOS DE RNA
# ==============================================================================

class PerceptronSimples:
    """Perceptron com funcao degrau bipolar e criterio de max epocas."""
    def __init__(self, lr=0.01, max_epochs=100):
        self.lr = lr
        self.max_epochs = max_epochs
        self.weights = None
        self.errors_per_epoch = []

    def fit(self, X, y):
        N, d = X.shape
        self.weights = np.random.uniform(-0.5, 0.5, d)
        self.errors_per_epoch = []
        for epoch in range(self.max_epochs):
            # Calcula predicoes e erros vetorialmente
            u_all = X @ self.weights
            y_pred_all = np.where(u_all >= 0, 1.0, -1.0)
            wrong = y_pred_all != y
            err_count = int(np.sum(wrong))
            # Atualiza pesos com base em amostras erradas (Perceptron batch)
            if err_count > 0:
                diff = y - y_pred_all  # (N,) so errados sao != 0
                # Atualizacao online simulada: update acumulado
                for i in np.where(wrong)[0]:
                    self.weights += self.lr * diff[i] * X[i]
            self.errors_per_epoch.append(err_count / N)
            if err_count == 0:
                break

    def predict(self, X):
        return np.where(X @ self.weights >= 0, 1.0, -1.0)


class ADALINE:
    """ADALINE com regra de aprendizado LMS (batch) e degrau bipolar na classificacao."""
    def __init__(self, lr=0.01, max_epochs=100, tol=1e-6):
        self.lr = lr
        self.max_epochs = max_epochs
        self.tol = tol
        self.weights = None
        self.errors_per_epoch = []

    def fit(self, X, y):
        N, d = X.shape
        self.weights = np.random.uniform(-0.5, 0.5, d)
        self.errors_per_epoch = []
        for epoch in range(self.max_epochs):
            u = X @ self.weights  # (N,)
            error = y - u  # (N,)
            eqm = np.mean(error ** 2)
            self.errors_per_epoch.append(eqm)
            # Batch gradient update
            grad = -(X.T @ error) / N  # (d,)
            self.weights -= self.lr * grad
            if epoch > 0 and abs(self.errors_per_epoch[-1] - self.errors_per_epoch[-2]) < self.tol:
                break

    def predict(self, X):
        return np.where(X @ self.weights >= 0, 1.0, -1.0)


class MLP:
    """MLP com backpropagation e mini-batch. Ativacao tanh ou sigmoid."""
    def __init__(self, hidden_layers=(10,), lr=0.01, max_epochs=200, tol=1e-6,
                 activation='tanh', batch_size=32):
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.max_epochs = max_epochs
        self.tol = tol
        self.activation = activation
        self.batch_size = batch_size
        self.W = []
        self.b = []
        self.errors_per_epoch = []

    def _act(self, z):
        return np.tanh(z) if self.activation == 'tanh' else 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _act_d(self, a):
        return 1.0 - a**2 if self.activation == 'tanh' else a * (1.0 - a)

    def fit(self, X, y):
        N, pin = X.shape
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        C = y.shape[1]

        layers = [pin] + list(self.hidden_layers) + [C]
        self.W = []
        self.b = []
        for i in range(len(layers) - 1):
            lim = np.sqrt(6.0 / (layers[i] + layers[i + 1]))
            self.W.append(np.random.uniform(-lim, lim, (layers[i], layers[i + 1])))
            self.b.append(np.zeros((1, layers[i + 1])))

        self.errors_per_epoch = []
        bs = self.batch_size

        for epoch in range(self.max_epochs):
            idx = np.random.permutation(N)
            epoch_err = 0.0

            for s in range(0, N, bs):
                bi = idx[s:s + bs]
                xb, yb = X[bi], y[bi]
                B = xb.shape[0]

                # Forward
                acts = [xb]
                for l in range(len(self.W)):
                    z = acts[-1] @ self.W[l] + self.b[l]
                    acts.append(self._act(z))

                out = acts[-1]
                err = yb - out
                epoch_err += np.sum(err ** 2)

                # Backward
                delta = err * self._act_d(out)
                for l in range(len(self.W) - 1, -1, -1):
                    self.W[l] += (self.lr / B) * (acts[l].T @ delta)
                    self.b[l] += (self.lr / B) * np.sum(delta, axis=0, keepdims=True)
                    if l > 0:
                        delta = (delta @ self.W[l].T) * self._act_d(acts[l])

            eqm = epoch_err / (2 * N)
            self.errors_per_epoch.append(eqm)
            if epoch > 0 and abs(self.errors_per_epoch[-1] - self.errors_per_epoch[-2]) < self.tol:
                break

    def predict(self, X):
        a = X
        for l in range(len(self.W)):
            a = self._act(a @ self.W[l] + self.b[l])
        if a.shape[1] == 1:
            return np.where(a.flatten() >= 0, 1.0, -1.0)
        return a



# ==============================================================================
# 6. ANALISE DE UNDERFITTING E OVERFITTING (MLP e RBF)
# ==============================================================================

print("\n=== ANALISE DE UNDERFITTING E OVERFITTING ===")

X_norm, x_min, x_max = normalize_minmax(X_raw)
X_bias = add_bias(X_norm)
y = y_raw.copy()

np.random.seed(42)
idx_all = np.random.permutation(N_total)
n_train = int(0.8 * N_total)
train_idx, test_idx = idx_all[:n_train], idx_all[n_train:]

X_train_b, X_test_b = X_bias[train_idx], X_bias[test_idx]
X_train, X_test = X_norm[train_idx], X_norm[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# --- MLP ---
mlp_configs = {
    'underfitting': {'hidden_layers': (2,), 'lr': 0.01, 'max_epochs': 50, 'batch_size': 64},
    'adequado':     {'hidden_layers': (20, 10), 'lr': 0.01, 'max_epochs': 200, 'batch_size': 32},
    'overfitting':  {'hidden_layers': (80, 40, 20), 'lr': 0.01, 'max_epochs': 300, 'batch_size': 16},
}

print("\n--- MLP ---")
for cname, params in mlp_configs.items():
    np.random.seed(42)
    t0 = time.time()
    mlp = MLP(**params, activation='tanh')
    mlp.fit(X_train, y_train.reshape(-1, 1))

    yp_tr = mlp.predict(X_train)
    yp_te = mlp.predict(X_test)
    m_tr = binary_metrics(y_train, yp_tr)
    m_te = binary_metrics(y_test, yp_te)

    print(f"\n  MLP [{cname}] - Topologia: {params['hidden_layers']} ({time.time()-t0:.1f}s)")
    print(f"    Treino -> Acc: {m_tr['acuracia']:.4f}")
    print(f"    Teste  -> Acc: {m_te['acuracia']:.4f}, Sens: {m_te['sensibilidade']:.4f}, "
          f"Spec: {m_te['especificidade']:.4f}")

    plot_curve(mlp.errors_per_epoch, f"MLP {cname} - Curva de Aprendizado",
               os.path.join(RESULTS_DIR, f"02_mlp_{cname}_curva.png"))
    cm = confusion_matrix_manual(y_test, yp_te, [-1, 1])
    plot_cm(cm, ['Classe -1', 'Classe +1'], f"MLP {cname}",
            os.path.join(RESULTS_DIR, f"02_mlp_{cname}_cm.png"))


# ==============================================================================
# 7. VALIDACAO POR AMOSTRAGEM ALEATORIA - R=500 RODADAS
# ==============================================================================

print("\n=== VALIDACAO MONTE CARLO (R=500) ===")
R = 500
metrics_names = ['acuracia', 'sensibilidade', 'especificidade', 'precisao', 'f1_score']

model_configs = {
    'Perceptron Simples': lambda: PerceptronSimples(lr=0.01, max_epochs=50),
    'ADALINE':            lambda: ADALINE(lr=0.1, max_epochs=50, tol=1e-6),
    'MLP':                lambda: MLP(hidden_layers=(20, 10), lr=0.05, max_epochs=100,
                                      activation='tanh', batch_size=64),
}

results = {name: {m: [] for m in metrics_names} for name in model_configs}
bw_data = {name: {'best_acc': -1, 'worst_acc': 2,
                   'best_r': None, 'worst_r': None,
                   'best_cm': None, 'worst_cm': None,
                   'best_err': None, 'worst_err': None}
           for name in model_configs}

t_start = time.time()
for r in range(R):
    if (r + 1) % 50 == 0:
        elapsed = time.time() - t_start
        print(f"  Rodada {r+1}/{R} ({elapsed:.0f}s total)")

    idx_perm = np.random.permutation(N_total)
    n_tr = int(0.8 * N_total)
    tr_i, te_i = idx_perm[:n_tr], idx_perm[n_tr:]

    X_tr_n, xmn, xmx = normalize_minmax(X_raw[tr_i])
    X_te_n, _, _ = normalize_minmax(X_raw[te_i], xmn, xmx)
    X_tr_b = add_bias(X_tr_n)
    X_te_b = add_bias(X_te_n)
    y_tr, y_te = y_raw[tr_i], y_raw[te_i]

    for mname, mfn in model_configs.items():
        model = mfn()
        if mname in ['Perceptron Simples', 'ADALINE']:
            model.fit(X_tr_b, y_tr)
            yp = model.predict(X_te_b)
        else:
            model.fit(X_tr_n, y_tr.reshape(-1, 1))
            yp = model.predict(X_te_n)

        m = binary_metrics(y_te, yp)
        for metric in metrics_names:
            results[mname][metric].append(m[metric])

        acc = m['acuracia']
        bw = bw_data[mname]
        if acc > bw['best_acc']:
            bw['best_acc'] = acc
            bw['best_r'] = r
            bw['best_cm'] = confusion_matrix_manual(y_te, yp, [-1, 1])
            bw['best_err'] = model.errors_per_epoch.copy()
        if acc < bw['worst_acc']:
            bw['worst_acc'] = acc
            bw['worst_r'] = r
            bw['worst_cm'] = confusion_matrix_manual(y_te, yp, [-1, 1])
            bw['worst_err'] = model.errors_per_epoch.copy()

print(f"Monte Carlo concluido! Tempo total: {time.time()-t_start:.0f}s")


# ==============================================================================
# 8. MATRIZES DE CONFUSAO E CURVAS (MELHOR E PIOR POR METRICA)
# ==============================================================================

print("\n=== MELHOR E PIOR RODADA POR MODELO (ACURACIA) ===")
for mname in model_configs:
    bw = bw_data[mname]
    print(f"\n{mname}:")
    print(f"  Melhor: {bw['best_acc']:.4f} (rodada {bw['best_r']})")
    print(f"  Pior:   {bw['worst_acc']:.4f} (rodada {bw['worst_r']})")

    sn = mname.replace(' ', '_')
    plot_cm(bw['best_cm'], ['Classe -1', 'Classe +1'],
            f"{mname} - Melhor Acc ({bw['best_acc']:.4f})",
            os.path.join(RESULTS_DIR, f"03_{sn}_melhor_cm.png"))
    plot_cm(bw['worst_cm'], ['Classe -1', 'Classe +1'],
            f"{mname} - Pior Acc ({bw['worst_acc']:.4f})",
            os.path.join(RESULTS_DIR, f"03_{sn}_pior_cm.png"))

    if bw['best_err']:
        plot_curve(bw['best_err'], f"{mname} - Curva (Melhor)",
                   os.path.join(RESULTS_DIR, f"03_{sn}_melhor_curva.png"))
    if bw['worst_err']:
        plot_curve(bw['worst_err'], f"{mname} - Curva (Pior)",
                   os.path.join(RESULTS_DIR, f"03_{sn}_pior_curva.png"))

# Tambem extrair melhor/pior por CADA metrica (item 6)
print("\n=== MELHOR E PIOR POR CADA METRICA ===")
for metric in metrics_names:
    print(f"\n--- {metric} ---")
    for mname in model_configs:
        vals = results[mname][metric]
        best_r = int(np.argmax(vals))
        worst_r = int(np.argmin(vals))
        print(f"  {mname}: melhor={vals[best_r]:.4f} (r={best_r}), pior={vals[worst_r]:.4f} (r={worst_r})")


# ==============================================================================
# 9. TABELAS E GRAFICOS FINAIS
# ==============================================================================

print("\n=== TABELAS DE RESULTADOS (500 RODADAS) ===")
for metric in metrics_names:
    print(f"\n--- {metric.upper()} ---")
    print(f"{'Modelo':<30} {'Media':>10} {'Desvio':>10} {'Maior':>10} {'Menor':>10}")
    print("-" * 72)
    for mname in model_configs:
        v = np.array(results[mname][metric])
        print(f"{mname:<30} {v.mean():>10.4f} {v.std():>10.4f} {v.max():>10.4f} {v.min():>10.4f}")

# Boxplots (uma figura com todas as metricas)
fig, axes = plt.subplots(1, len(metrics_names), figsize=(22, 5))
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
for i, metric in enumerate(metrics_names):
    dbox = [np.array(results[n][metric]) for n in model_configs]
    bp = axes[i].boxplot(dbox, tick_labels=[n.replace(' ', '\n') for n in model_configs], patch_artist=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    axes[i].set_title(metric.replace('_', ' ').title())
    axes[i].grid(True, alpha=0.3)
    axes[i].tick_params(axis='x', labelsize=7)
plt.suptitle("Boxplot - Desempenho dos Modelos (500 Rodadas, Spiral)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "04_boxplots_metricas.png"), dpi=150, bbox_inches='tight')
plt.close()

# Violin plots
fig, axes = plt.subplots(1, len(metrics_names), figsize=(22, 5))
for i, metric in enumerate(metrics_names):
    dv = [np.array(results[n][metric]) for n in model_configs]
    axes[i].violinplot(dv, showmeans=True, showmedians=True)
    axes[i].set_xticks(range(1, len(model_configs) + 1))
    axes[i].set_xticklabels([n.replace(' ', '\n') for n in model_configs], fontsize=7)
    axes[i].set_title(metric.replace('_', ' ').title())
    axes[i].grid(True, alpha=0.3)
plt.suptitle("Violin Plot - Desempenho dos Modelos (500 Rodadas, Spiral)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "04_violinplots_metricas.png"), dpi=150, bbox_inches='tight')
plt.close()

print(f"\nTodos os resultados salvos em: {RESULTS_DIR}")
print("Etapa 1 concluida com sucesso!")
