import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings('ignore')

# Установка русского шрифта для графиков
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


# ==================== 1. ГЕНЕРАЦИЯ ДАННЫХ ====================

def generate_data():
    """
    Генерация данных на основе центров классов из лабораторных №2 и №3
    Данные из лаб2 были 3D -> отбрасываем 3-е измерение (по заданию)
    """
    np.random.seed(42)  # Для воспроизводимости

    # Определение центров классов (5 классов)
    centers = np.array([
        [-5.0, 2.0],  # Класс 1 (из лаб2, без 3-й координаты)
        [-4.0, -5.0],  # Класс 2 (из лаб2, без 3-й координаты)
        [4.0, -2.0],  # Класс 3 (из лаб3)
        [3.0, 2.0],  # Класс 4 (из лаб3)
        [4.0, 1.0]  # Класс 5 (из лаб3)
    ])

    # Ковариационные матрицы для каждого класса
    cov_matrices = [
        [[11.0, -0.8], [-0.8, 10.0]],  # Ковариация для класса 1 (из лаб2, без 3D)
        [[11.0, -0.8], [-0.8, 10.0]],  # Ковариация для класса 2
        [[2.0, -0.2], [-0.2, 1.0]],  # Ковариация для класса 3 (из лаб3)
        [[3.0, -1.0], [-1.0, 3.0]],  # Ковариация для класса 4 (из лаб3)
        [[3.0, 1.5], [1.5, 1.0]]  # Ковариация для класса 5 (из лаб3)
    ]

    N_per_class = 50  # Количество точек в каждом классе
    N_total = N_per_class * 5
    X = []
    y_true = []

    for i in range(5):
        points = np.random.multivariate_normal(centers[i], cov_matrices[i], N_per_class)
        X.append(points)
        y_true.extend([i] * N_per_class)

    X = np.vstack(X)
    y_true = np.array(y_true)

    return X, y_true, centers, N_per_class


# ==================== 2. ФУНКЦИИ ДЛЯ ОЦЕНКИ КАЧЕСТВА ====================

def calculate_error_rate(y_true, y_pred, N_per_class):
    """
    Расчёт частотности ошибок кластеризации
    """
    M = len(np.unique(y_true))
    N = len(y_true)

    # Создаём матрицу соответствия
    confusion = np.zeros((M, M))
    for i in range(N):
        confusion[y_true[i], y_pred[i]] += 1

    # Находим наилучшее соответствие (венгерский алгоритм упрощённо)
    from scipy.optimize import linear_sum_assignment
    cost = -confusion  # Максимизируем совпадения
    row_ind, col_ind = linear_sum_assignment(cost)

    correct = sum(confusion[row_ind[i], col_ind[i]] for i in range(M))
    error_rate = (N - correct) / N

    # Создаём отображение предсказанных кластеров в истинные
    mapping = {col_ind[i]: row_ind[i] for i in range(M)}
    y_pred_mapped = np.array([mapping[p] for p in y_pred])

    return error_rate, y_pred_mapped


def silhouette_plot(X, labels, title):
    """
    Построение графика коэффициента силуэта
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)

    n_clusters = len(np.unique(labels))
    y_lower = 10

    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    for i in range(n_clusters):
        cluster_values = sample_silhouette_values[labels == i]
        cluster_values.sort()
        size = len(cluster_values)
        y_upper = y_lower + size

        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_values,
                         facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, str(i + 1))
        y_lower = y_upper + 10

    ax.axvline(x=silhouette_avg, color="red", linestyle="--",
               label=f'Средний силуэт: {silhouette_avg:.3f}')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Значения коэффициента силуэта", fontsize=12)
    ax.set_ylabel("Метка кластера", fontsize=12)
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, y_lower])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_clustering_results(X, y_true, y_pred, centers, title, save_name=None):
    """
    Визуализация результатов кластеризации
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Истинная разметка
    colors_true = ['red', 'blue', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']

    for i in range(5):
        mask = y_true == i
        ax1.scatter(X[mask, 0], X[mask, 1], c=colors_true[i],
                    marker=markers[i], s=60, alpha=0.7, label=f'Класс {i + 1}')
    ax1.set_title("Истинное распределение классов", fontsize=12)
    ax1.set_xlabel("Признак 1")
    ax1.set_ylabel("Признак 2")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Результат кластеризации
    for i in range(5):
        mask = y_pred == i
        ax2.scatter(X[mask, 0], X[mask, 1], c=colors_true[i],
                    marker=markers[i], s=60, alpha=0.7, label=f'Кластер {i + 1}')
    ax2.set_title(title, fontsize=12)
    ax2.set_xlabel("Признак 1")
    ax2.set_ylabel("Признак 2")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ==================== 3. K-MEANS КЛАСТЕРИЗАЦИЯ ====================

def kmeans_clustering(X, n_clusters=5):
    """
    Кластеризация методом k-means с разными метриками
    """
    results = {}
    metrics = {
        'euclidean': 'sqeuclidean',
        'cityblock': 'manhattan',
        'cosine': 'cosine'
    }
    metric_names = {
        'euclidean': 'Евклидово расстояние',
        'cityblock': 'Манхэттенское расстояние',
        'cosine': 'Косинусное расстояние'
    }

    print("\n" + "=" * 60)
    print("АЛГОРИТМ K-MEANS")
    print("=" * 60)

    for metric, km_metric in metrics.items():
        print(f"\n--- Метрика: {metric_names[metric]} ---")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42,
                        n_init=10, max_iter=300)
        y_pred = kmeans.fit_predict(X)

        error_rate, y_pred_mapped = calculate_error_rate(y_true, y_pred, N_per_class)
        print(f"Частотность ошибок: {error_rate * 100:.2f}%")

        # Силуэт-график
        fig_sil = silhouette_plot(X, y_pred,
                                  f'Коэффициент силуэта (k-means, {metric_names[metric]})')

        # Визуализация
        fig_clust = plot_clustering_results(X, y_true, y_pred_mapped,
                                            kmeans.cluster_centers_,
                                            f'Результат k-means ({metric_names[metric]})')

        results[metric] = {
            'error_rate': error_rate,
            'labels': y_pred,
            'centers': kmeans.cluster_centers_
        }

        plt.show()

    return results


# ==================== 4. ИЕРАРХИЧЕСКАЯ КЛАСТЕРИЗАЦИЯ ====================

def hierarchical_clustering(X, n_clusters=5):
    """
    Иерархическая кластеризация с разными метриками
    """
    results = {}
    linkage_methods = {
        'ward': 'Варда (евклидова)',
        'complete': 'Полная связь (евклидова)',
        'average': 'Средняя связь (евклидова)'
    }

    print("\n" + "=" * 60)
    print("ИЕРАРХИЧЕСКАЯ КЛАСТЕРИЗАЦИЯ")
    print("=" * 60)

    for method, method_name in linkage_methods.items():
        print(f"\n--- Метод: {method_name} ---")

        # Построение дендрограммы
        Z = linkage(X, method=method, metric='euclidean')

        fig_dend, ax = plt.subplots(figsize=(12, 6))
        dendrogram(Z, ax=ax, truncate_mode='lastp', p=30, leaf_rotation=90.)
        ax.set_title(f'Дендрограмма (метод {method_name})', fontsize=14)
        ax.set_xlabel('Индекс образца')
        ax.set_ylabel('Расстояние')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Кластеризация
        y_pred = fcluster(Z, n_clusters, criterion='maxclust') - 1

        error_rate, y_pred_mapped = calculate_error_rate(y_true, y_pred, N_per_class)
        print(f"Частотность ошибок: {error_rate * 100:.2f}%")

        # Визуализация результатов
        fig_clust = plot_clustering_results(X, y_true, y_pred_mapped, None,
                                            f'Результат иерархической кластеризации ({method_name})')

        results[method] = {
            'error_rate': error_rate,
            'labels': y_pred,
            'linkage': Z
        }

        plt.show()

    return results


# ==================== 5. ОСНОВНАЯ ПРОГРАММА ====================

if __name__ == "__main__":
    print("=" * 70)
    print("ЛАБОРАТОРНАЯ РАБОТА №8")
    print("Исследование алгоритмов кластеризации")
    print("=" * 70)

    # Генерация данных
    print("\n1. Генерация данных на основе центров из лаб №2 и №3...")
    X, y_true, centers, N_per_class = generate_data()
    print(f"   - Сгенерировано {len(X)} точек")
    print(f"   - Количество кластеров: {len(np.unique(y_true))}")
    print(f"   - Центры кластеров:")
    for i, center in enumerate(centers):
        print(f"     Класс {i + 1}: {center}")

    # Показываем исходные данные
    fig_init, ax = plt.subplots(figsize=(10, 8))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    for i in range(5):
        mask = y_true == i
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], marker=markers[i],
                   s=50, alpha=0.6, label=f'Класс {i + 1}')
    ax.set_title("Исходные данные для кластеризации", fontsize=14)
    ax.set_xlabel("Признак 1")
    ax.set_ylabel("Признак 2")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

    # K-means кластеризация
    kmeans_results = kmeans_clustering(X)

    # Иерархическая кластеризация
    hier_results = hierarchical_clustering(X)

    # ==================== 6. СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ====================
    print("\n" + "=" * 60)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 60)

    print("\nK-MEANS:")
    print("-" * 50)
    for metric, res in kmeans_results.items():
        print(f"{metric:12s}: частота ошибок = {res['error_rate'] * 100:6.2f}%")

    print("\nИЕРАРХИЧЕСКАЯ КЛАСТЕРИЗАЦИЯ:")
    print("-" * 50)
    for method, res in hier_results.items():
        print(f"{method:10s}: частота ошибок = {res['error_rate'] * 100:6.2f}%")

    # Определяем лучший метод
    best_kmeans = min(kmeans_results.items(), key=lambda x: x[1]['error_rate'])
    best_hier = min(hier_results.items(), key=lambda x: x[1]['error_rate'])

    print("\n" + "=" * 60)
    print("ВЫВОДЫ")
    print("=" * 60)
    print(f"\nЛучшая метрика для k-means: {best_kmeans[0]}")
    print(f"Минимальная частота ошибок: {best_kmeans[1]['error_rate'] * 100:.2f}%")
    print(f"\nЛучший метод иерархической кластеризации: {best_hier[0]}")
    print(f"Минимальная частота ошибок: {best_hier[1]['error_rate'] * 100:.2f}%")