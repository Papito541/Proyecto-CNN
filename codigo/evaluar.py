
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

BASE = Path(__file__).resolve().parent.parent
DATOS = BASE / "data" / "modelado_9_clases"
MODELO = BASE / "resultados" / "cnn_9_clases" / "mejor_modelo.keras"
SALIDA = BASE / "resultados" / "evaluacion_cnn2.0"
SALIDA.mkdir(parents=True, exist_ok=True)

print("=== EVALUACION CNN ===")

# 1. Cargar datos y modelo
X_test = np.load(
    DATOS / "X_test.npy",
    mmap_mode="r"
)
y_test = np.load(
    DATOS / "y_test.npy"
)

with open(
    DATOS / "configuracion.json",
    encoding="utf-8"
) as f:
    config = json.load(f)

clases = config["clases"]
n_clases = len(clases)

modelo = tf.keras.models.load_model(MODELO)

X_test = X_test.reshape(-1, 65, 1)

# 2. Calentamiento para reducir efecto de inicialización
print("\nPreparando inferencia...")
modelo.predict(X_test[:512], verbose=0)

# 3. Predicciones y latencia
print("Generando predicciones...")

inicio = perf_counter()

probabilidades = modelo.predict(
    X_test,
    batch_size=512,
    verbose=1
)

tiempo_total = perf_counter() - inicio

y_pred = np.argmax(
    probabilidades, axis=1
)

latencia_ms = (
    tiempo_total / len(X_test)
) * 1000

# 4. Metricas generales
accuracy = accuracy_score(y_test, y_pred)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )
)

print("\n=== METRICAS GENERALES ===")
print(f"Accuracy: {accuracy:.6f}")
print(f"Precision macro: {precision:.6f}")
print(f"Recall macro: {recall:.6f}")
print(f"F1-score macro: {f1:.6f}")

# 5. Metricas por clase
reporte = classification_report(
    y_test,
    y_pred,
    labels=np.arange(n_clases),
    target_names=clases,
    output_dict=True,
    zero_division=0
)

df_reporte = pd.DataFrame(reporte).T

df_reporte.to_csv(
    SALIDA / "metricas_por_clase.csv"
)

print("\n=== METRICAS POR CLASE ===")
print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(n_clases),
        target_names=clases,
        zero_division=0
    )
)

# 6. Matriz de confusion
matriz = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(n_clases)
)

pd.DataFrame(
    matriz,
    index=clases,
    columns=clases
).to_csv(
    SALIDA / "matriz_confusion.csv"
)

fig, ax = plt.subplots(figsize=(14, 12))

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=clases
)

disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=90,
    values_format="d",
    colorbar=False
)

plt.tight_layout()
plt.savefig(
    SALIDA / "matriz_confusion.png",
    dpi=200
)
plt.close()

# 7. Tiempo de inferencia
print("\n=== TIEMPO DE INFERENCIA ===")
print(f"Registros evaluados: {len(X_test):,}")
print(f"Tiempo total: {tiempo_total:.4f} s")
print(f"Promedio por registro: {latencia_ms:.6f} ms")
print(
    "Rendimiento:",
    f"{len(X_test) / tiempo_total:.2f} registros/s"
)

# 8. Guardar resumen
resumen = {
    "accuracy": float(accuracy),
    "precision_macro": float(precision),
    "recall_macro": float(recall),
    "f1_macro": float(f1),
    "tiempo_total_segundos": float(tiempo_total),
    "latencia_media_ms_por_registro": float(latencia_ms),
    "registros_evaluados": int(len(X_test)),
    "batch_size": 512
}

with open(
    SALIDA / "resumen_evaluacion.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(resumen, f, indent=2)

print("\n=== EVALUACION FINALIZADA ===")
print("Resultados:", SALIDA)
