"""Линейная регрессия: реализация с нуля и через sklearn."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# === 1. Генерация данных ===
np.random.seed(42)
n = 200
area = np.random.uniform(30, 200, n)
rooms = np.random.randint(1, 6, n)
price = 50 * area + 20 * rooms + np.random.normal(0, 500, n)

X = np.column_stack([area, rooms])
y = price

# === 2. Реализация МНК с нуля ===
def ols_fit(X, y):
    """Метод наименьших квадратов: beta = (X^T X)^-1 X^T y."""
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # добавляем bias
    beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return beta

def ols_predict(X, beta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return X_b @ beta

beta = ols_fit(X, y)
y_pred_manual = ols_predict(X, beta)
print("Коэффициенты (с нуля):", beta)

# === 3. Сравнение со sklearn ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"sklearn coef: {model.coef_}, intercept: {model.intercept_}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# === 4. Визуализация остатков ===
residuals = y_test - y_pred
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Предсказание")
plt.ylabel("Остатки")
plt.title("Анализ остатков")
plt.savefig("residuals.png")
plt.show()

# === 5. Сохранение модели ===
joblib.dump(model, "linear_model.pkl")
print("Модель сохранена")
