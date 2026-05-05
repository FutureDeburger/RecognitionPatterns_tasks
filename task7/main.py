"""
Лабораторная работа №7
Распознавание образов с использованием машины опорных векторов (SVM)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Настройка русских шрифтов (раскомментировать при необходимости)
# plt.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================================
# ЗАДАНИЕ 1: Линейно разделимые выборки (из лабы 2)
# ============================================================================
print("=" * 60)
print("ЗАДАНИЕ 1: Линейно разделимые выборки")
print("=" * 60)

# Данные из лабы 2 (отбрасываем третье измерение)
# Класс 1
m1_3d = np.array([-5, 2, -3])
m1 = m1_3d[:2]  # [-5, 2]

# Класс 2
m2_3d = np.array([-4, -5, 4])
m2 = m2_3d[:2]  # [-4, -5]

# Ковариационная матрица (берём первые 2x2)
C_full = np.array([[11, -0.8, 1.0],
                   [-0.8, 10, 0.5],
                   [1.0, 0.5, 7]])
C = C_full[:2, :2]  # [[11, -0.8], [-0.8, 10]]

print(f"Класс 1: мат. ожидание = {m1}")
print(f"Класс 2: мат. ожидание = {m2}")
print(f"Ковариационная матрица:\n{C}")

# Параметры эксперимента
n = 2  # размерность
M = 2  # число классов
K = 500  # количество испытаний
train_size = 200  # размер обучающей выборки на класс

# Генерация данных
np.random.seed(42)

# Обучающая выборка
X_train_class1 = np.random.multivariate_normal(m1, C, train_size)
X_train_class2 = np.random.multivariate_normal(m2, C, train_size)
X_train = np.vstack([X_train_class1, X_train_class2])
y_train = np.hstack([np.zeros(train_size), np.ones(train_size)])

# Тестовая выборка
X_test_class1 = np.random.multivariate_normal(m1, C, K)
X_test_class2 = np.random.multivariate_normal(m2, C, K)
X_test = np.vstack([X_test_class1, X_test_class2])
y_test = np.hstack([np.zeros(K), np.ones(K)])

print(f"\nОбучающая выборка: {X_train.shape[0]} образцов")
print(f"Тестовая выборка: {X_test.shape[0]} образцов")

# Визуализация данных
plt.figure(figsize=(10, 8))
plt.scatter(X_train_class1[:, 0], X_train_class1[:, 1], c='blue', alpha=0.6, label='Класс 1')
plt.scatter(X_train_class2[:, 0], X_train_class2[:, 1], c='red', alpha=0.6, label='Класс 2')
plt.scatter(m1[0], m1[1], c='darkblue', marker='*', s=200, label='Центр класса 1')
plt.scatter(m2[0], m2[1], c='darkred', marker='*', s=200, label='Центр класса 2')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.title('Линейно разделимые классы (лаба 2, отброшено 3-е измерение)')
plt.legend()
plt.grid(alpha=0.3)
plt.axis('equal')
plt.show()

# ----------------------------------------------------------------------------
# Эксперимент 1.1: Перебор параметра регуляризации C
# ----------------------------------------------------------------------------
print("\n" + "-" * 40)
print("Эксперимент 1.1: Перебор параметра регуляризации C")
print("-" * 40)

C_values = np.logspace(-3, 3, 15)  # от 0.001 до 1000
train_accuracies = []
test_accuracies = []

for C_val in C_values:
    svm_model = svm.SVC(kernel='linear', C=C_val, random_state=42)
    svm_model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, svm_model.predict(X_train))
    test_acc = accuracy_score(y_test, svm_model.predict(X_test))

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

# Поиск оптимального C
best_idx = np.argmax(test_accuracies)
best_C = C_values[best_idx]
best_acc = test_accuracies[best_idx]

print(f"Оптимальное значение C = {best_C:.4f}")
print(f"Максимальная точность на тесте = {best_acc:.4f} ({best_acc * 100:.2f}%)")

# График зависимости точности от C
plt.figure(figsize=(10, 6))
plt.semilogx(C_values, train_accuracies, 'b-o', label='Обучающая выборка')
plt.semilogx(C_values, test_accuracies, 'r-s', label='Тестовая выборка')
plt.axvline(x=best_C, color='g', linestyle='--', label=f'Оптимальное C = {best_C:.4f}')
plt.xlabel('Параметр регуляризации C (логарифмическая шкала)')
plt.ylabel('Точность распознавания')
plt.title('Зависимость точности от параметра C (линейное ядро)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ----------------------------------------------------------------------------
# Обучение SVM с оптимальным C и вычисление матрицы ошибок
# ----------------------------------------------------------------------------
print("\n" + "-" * 40)
print("Финальное обучение SVM с оптимальным C")
print("-" * 40)

svm_best = svm.SVC(kernel='linear', C=best_C, random_state=42)
svm_best.fit(X_train, y_train)

# Предсказания
y_train_pred = svm_best.predict(X_train)
y_test_pred = svm_best.predict(X_test)

# Матрицы ошибок
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

# Вероятности ошибок (ошибка = 1 - точность)
error_train = 1 - accuracy_score(y_train, y_train_pred)
error_test = 1 - accuracy_score(y_test, y_test_pred)

print(f"\nРезультаты SVM (линейное ядро, C={best_C:.4f}):")
print(f"  Ошибка на обучении: {error_train:.4f} ({error_train * 100:.2f}%)")
print(f"  Ошибка на тесте:   {error_test:.4f} ({error_test * 100:.2f}%)")

print(f"\nМатрица ошибок (тест):")
print(f"            Предсказан класс 1    Предсказан класс 2")
print(f"Класс 1:        {cm_test[0, 0]:4d}                 {cm_test[0, 1]:4d}")
print(f"Класс 2:        {cm_test[1, 0]:4d}                 {cm_test[1, 1]:4d}")


# ----------------------------------------------------------------------------
# Теоретическая матрица ошибок (гауссовский классификатор)
# ----------------------------------------------------------------------------
def compute_theoretical_errors(m1, m2, C1, C2, pw1, pw2):
    """Расчёт теоретических вероятностей ошибок для двух классов"""
    n = len(m1)
    dm = m1 - m2
    C1_inv = np.linalg.inv(C1)
    C2_inv = np.linalg.inv(C2)

    l0 = np.log(pw2 / pw1)

    # Для класса 1
    tr1 = np.trace(np.eye(n) - C1_inv @ C2)
    tr1_2 = np.trace(np.linalg.matrix_power(np.eye(n) - C1_inv @ C2, 2))
    mg1 = 0.5 * (tr1 + dm.T @ C1_inv @ dm - np.log(np.linalg.det(C1) / np.linalg.det(C2)))
    Dg1 = 0.5 * tr1_2 + dm.T @ C1_inv @ C2 @ C1_inv @ dm
    P_err1 = norm.cdf(l0, loc=mg1, scale=np.sqrt(Dg1))

    # Для класса 2
    tr2 = np.trace(C2_inv @ C1 - np.eye(n))
    tr2_2 = np.trace(np.linalg.matrix_power(C2_inv @ C1 - np.eye(n), 2))
    mg2 = 0.5 * (tr2 + dm.T @ C2_inv @ dm - np.log(np.linalg.det(C2) / np.linalg.det(C1)))
    Dg2 = 0.5 * tr2_2 + dm.T @ C2_inv @ C1 @ C2_inv @ dm
    P_err2 = 1 - norm.cdf(l0, loc=mg2, scale=np.sqrt(Dg2))

    return np.array([[1 - P_err1, P_err1], [P_err2, 1 - P_err2]])


# Априорные вероятности (равные)
pw = np.array([0.5, 0.5])

P_theoretical = compute_theoretical_errors(m1, m2, C, C, pw[0], pw[1])

print(f"\nТеоретическая матрица вероятностей ошибок (гауссовский классификатор):")
print(f"            Класс 1       Класс 2")
print(f"Класс 1:    {P_theoretical[0, 0]:.4f}        {P_theoretical[0, 1]:.4f}")
print(f"Класс 2:    {P_theoretical[1, 0]:.4f}        {P_theoretical[1, 1]:.4f}")


# ----------------------------------------------------------------------------
# Визуализация решающей границы и опорных векторов
# ----------------------------------------------------------------------------
def plot_decision_boundary(model, X, y, title):
    """Визуализация решающей границы SVM"""
    plt.figure(figsize=(10, 8))

    # Создание сетки
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Решающая граница
    plt.contourf(xx, yy, Z, alpha=0.3, colors=['blue', 'red'])
    plt.contour(xx, yy, Z, colors='black', linewidths=1)

    # Данные
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', alpha=0.6, edgecolors='k', label='Класс 1')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', alpha=0.6, edgecolors='k', label='Класс 2')

    # Опорные векторы
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                s=150, facecolors='none', edgecolors='green', linewidths=2,
                label='Опорные векторы')

    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


plot_decision_boundary(svm_best, X_train, y_train,
                       f'Решающая граница SVM (линейное ядро, C={best_C:.4f})')

# ============================================================================
# ЗАДАНИЕ 2: Линейно НЕразделимые выборки (из лабы 3)
# ============================================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 2: Линейно НЕразделимые выборки (3 класса)")
print("=" * 60)

# Данные из лабы 3
m3_1 = np.array([4, -2])  # класс 1
m3_2 = np.array([3, 2])  # класс 2
m3_3 = np.array([4, 1])  # класс 3

C1 = np.array([[2, -0.2], [-0.2, 1]])
C2 = np.array([[3, -1.0], [-1.0, 3]])
C3 = np.array([[3, 1.5], [1.5, 1]])

print("Параметры классов (лаба 3):")
print(f"Класс 1: m={m3_1}, C1=\n{C1}")
print(f"Класс 2: m={m3_2}, C2=\n{C2}")
print(f"Класс 3: m={m3_3}, C3=\n{C3}")

# Параметры эксперимента
n = 2
M3 = 3
K = 500
train_size = 200

# Генерация данных
np.random.seed(42)

X_train_3 = []
y_train_3 = []
X_test_3 = []
y_test_3 = []

for i, (m, cov) in enumerate([(m3_1, C1), (m3_2, C2), (m3_3, C3)]):
    X_train_i = np.random.multivariate_normal(m, cov, train_size)
    X_test_i = np.random.multivariate_normal(m, cov, K)
    X_train_3.append(X_train_i)
    X_test_3.append(X_test_i)
    y_train_3.append(np.full(train_size, i))
    y_test_3.append(np.full(K, i))

X_train_3 = np.vstack(X_train_3)
y_train_3 = np.hstack(y_train_3)
X_test_3 = np.vstack(X_test_3)
y_test_3 = np.hstack(y_test_3)

print(f"\nОбучающая выборка: {X_train_3.shape[0]} образцов")
print(f"Тестовая выборка: {X_test_3.shape[0]} образцов")

# Визуализация
plt.figure(figsize=(10, 8))
colors = ['blue', 'green', 'orange']
labels = ['Класс 1', 'Класс 2', 'Класс 3']
for i in range(3):
    mask = y_train_3 == i
    plt.scatter(X_train_3[mask, 0], X_train_3[mask, 1],
                c=colors[i], alpha=0.5, label=labels[i])
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.title('Линейно НЕразделимые классы (лаба 3)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ----------------------------------------------------------------------------
# Эксперимент 2.1: Полиномиальное ядро - перебор степени полинома
# ----------------------------------------------------------------------------
print("\n" + "-" * 40)
print("Эксперимент 2.1: Полиномиальное ядро - перебор степени")
print("-" * 40)

degrees = range(1, 11)
poly_train_acc = []
poly_test_acc = []
C_fixed = 1.0  # фиксированный параметр регуляризации

for d in degrees:
    svm_poly = svm.SVC(kernel='poly', degree=d, C=C_fixed, random_state=42)
    svm_poly.fit(X_train_3, y_train_3)

    train_acc = accuracy_score(y_train_3, svm_poly.predict(X_train_3))
    test_acc = accuracy_score(y_test_3, svm_poly.predict(X_test_3))

    poly_train_acc.append(train_acc)
    poly_test_acc.append(test_acc)
    print(f"Степень {d}: точность на тесте = {test_acc:.4f}")

best_degree = degrees[np.argmax(poly_test_acc)]
print(f"\nОптимальная степень полинома = {best_degree}")

# График
plt.figure(figsize=(10, 6))
plt.plot(degrees, poly_train_acc, 'b-o', label='Обучающая выборка')
plt.plot(degrees, poly_test_acc, 'r-s', label='Тестовая выборка')
plt.axvline(x=best_degree, color='g', linestyle='--', label=f'Оптимальная степень = {best_degree}')
plt.xlabel('Степень полинома')
plt.ylabel('Точность распознавания')
plt.title('Зависимость точности от степени полиномиального ядра')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ----------------------------------------------------------------------------
# Эксперимент 2.2: RBF ядро - перебор масштаба (gamma)
# ----------------------------------------------------------------------------
print("\n" + "-" * 40)
print("Эксперимент 2.2: RBF ядро - перебор масштаба gamma")
print("-" * 40)

gamma_values = np.logspace(-3, 2, 15)  # от 0.001 до 100
rbf_train_acc = []
rbf_test_acc = []

for gamma in gamma_values:
    svm_rbf = svm.SVC(kernel='rbf', gamma=gamma, C=C_fixed, random_state=42)
    svm_rbf.fit(X_train_3, y_train_3)

    train_acc = accuracy_score(y_train_3, svm_rbf.predict(X_train_3))
    test_acc = accuracy_score(y_test_3, svm_rbf.predict(X_test_3))

    rbf_train_acc.append(train_acc)
    rbf_test_acc.append(test_acc)
    print(f"gamma = {gamma:.6f}: точность на тесте = {test_acc:.4f}")

best_gamma = gamma_values[np.argmax(rbf_test_acc)]
print(f"\nОптимальное значение gamma = {best_gamma:.6f}")

# График
plt.figure(figsize=(10, 6))
plt.semilogx(gamma_values, rbf_train_acc, 'b-o', label='Обучающая выборка')
plt.semilogx(gamma_values, rbf_test_acc, 'r-s', label='Тестовая выборка')
plt.axvline(x=best_gamma, color='g', linestyle='--', label=f'Оптимальная gamma = {best_gamma:.6f}')
plt.xlabel('Параметр масштаба gamma (логарифмическая шкала)')
plt.ylabel('Точность распознавания')
plt.title('Зависимость точности от gamma для RBF-ядра')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ----------------------------------------------------------------------------
# Финальное обучение с оптимальными параметрами
# ----------------------------------------------------------------------------
print("\n" + "-" * 40)
print("Финальное обучение с оптимальными параметрами")
print("-" * 40)

# Полиномиальное ядро
svm_poly_best = svm.SVC(kernel='poly', degree=best_degree, C=C_fixed, random_state=42)
svm_poly_best.fit(X_train_3, y_train_3)
y_pred_poly = svm_poly_best.predict(X_test_3)
error_poly = 1 - accuracy_score(y_test_3, y_pred_poly)

# RBF ядро
svm_rbf_best = svm.SVC(kernel='rbf', gamma=best_gamma, C=C_fixed, random_state=42)
svm_rbf_best.fit(X_train_3, y_train_3)
y_pred_rbf = svm_rbf_best.predict(X_test_3)
error_rbf = 1 - accuracy_score(y_test_3, y_pred_rbf)

print(f"\nРезультаты для 3 классов:")
print(f"  Полиномиальное ядро (степень={best_degree}): ошибка = {error_poly:.4f} ({error_poly * 100:.2f}%)")
print(f"  RBF ядро (gamma={best_gamma:.6f}):            ошибка = {error_rbf:.4f} ({error_rbf * 100:.2f}%)")

# Матрицы ошибок
cm_poly = confusion_matrix(y_test_3, y_pred_poly)
cm_rbf = confusion_matrix(y_test_3, y_pred_rbf)

print(f"\nМатрица ошибок (полиномиальное ядро):")
print(f"            Класс1   Класс2   Класс3")
for i in range(3):
    print(f"Класс{i + 1}:     {cm_poly[i, 0]:4d}     {cm_poly[i, 1]:4d}     {cm_poly[i, 2]:4d}")

print(f"\nМатрица ошибок (RBF ядро):")
print(f"            Класс1   Класс2   Класс3")
for i in range(3):
    print(f"Класс{i + 1}:     {cm_rbf[i, 0]:4d}     {cm_rbf[i, 1]:4d}     {cm_rbf[i, 2]:4d}")


# ----------------------------------------------------------------------------
# Визуализация решающих границ для 3 классов
# ----------------------------------------------------------------------------
def plot_3class_decision_boundary(model, X, y, title, kernel_name):
    """Визуализация решающей границы для 3 классов"""
    plt.figure(figsize=(10, 8))

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['blue', 'green', 'orange'])
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

    for i in range(3):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.6,
                    edgecolors='k', label=f'Класс {i + 1}')

    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                s=150, facecolors='none', edgecolors='black', linewidths=1.5,
                label='Опорные векторы')

    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


plot_3class_decision_boundary(svm_poly_best, X_train_3, y_train_3,
                              f'Решающая граница (полиномиальное ядро, степень={best_degree})',
                              'poly')

plot_3class_decision_boundary(svm_rbf_best, X_train_3, y_train_3,
                              f'Решающая граница (RBF ядро, gamma={best_gamma:.6f})',
                              'rbf')

# ============================================================================
# ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ
# ============================================================================
print("\n" + "=" * 60)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 60)

print("\n| Задание | Тип данных | Ядро | Параметр | Ошибка |")
print("|---------|------------|------|----------|--------|")
print(f"| 1       | Линейно разделимые | linear | C={best_C:.4f} | {error_test:.4f} |")
print(f"| 2       | Линейно неразделимые | polynomial | degree={best_degree} | {error_poly:.4f} |")
print(f"| 2       | Линейно неразделимые | RBF | gamma={best_gamma:.6f} | {error_rbf:.4f} |")

print("\n" + "=" * 60)
print("Программа завершена")
print("=" * 60)