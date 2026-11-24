
from pathlib import Path
import json
import gc

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

BASE = Path(__file__).resolve().parent.parent

ARCHIVO = (
    BASE / "data" / "processed" /
    "CICIDS2017_depurado.csv"
)

SALIDA = BASE / "data" / "modelado"
SALIDA.mkdir(parents=True, exist_ok=True)

SEMILLA = 42

# Variables constantes
CONSTANTES = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate"
]

# Redundancias exactas adicionales
REDUNDANTES = [
    "Subflow Fwd Packets",
    "Subflow Bwd Packets",
    "SYN Flag Count",
    "CWE Flag Count"
]

# Mediciones que no deberían ser negativas
INVALIDAS = [
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Min",
    "Fwd Header Length",
    "Bwd Header Length",
    "min_seg_size_forward"
]

META = ["archivo_origen", "fila_origen"]

print("=== PREPARACION CICIDS2017 ===")

# 1. Leer dataset
print("\n[1/7] Cargando dataset...")

df = pd.read_csv(ARCHIVO, low_memory=False)

print("Registros:", len(df))
print("Columnas:", len(df.columns))

# 2. Seleccionar variables
print("\n[2/7] Seleccionando caracteristicas...")

eliminar = CONSTANTES + REDUNDANTES

faltantes = set(eliminar + INVALIDAS + META + ["Label"]) - set(df.columns)

if faltantes:
    raise ValueError(f"Columnas faltantes: {faltantes}")

features = [
    c for c in df.columns
    if c not in eliminar + META + ["Label"]
]

assert len(features) == 65, (
    f"Se esperaban 65 caracteristicas, "
    f"pero hay {len(features)}"
)

print("Caracteristicas seleccionadas:", len(features))

# 3. Marcar anomalías
print("\n[3/7] Tratando valores invalidos...")

for columna in INVALIDAS:
    mascara = df[columna] < 0
    cantidad = int(mascara.sum())

    if cantidad:
        df.loc[mascara, columna] = np.nan

    print(f"{columna}: {cantidad} valores marcados")

# 4. Codificar etiquetas
print("\n[4/7] Codificando clases...")

encoder = LabelEncoder()
y = encoder.fit_transform(df["Label"])

print("Clases:", len(encoder.classes_))

for codigo, clase in enumerate(encoder.classes_):
    print(codigo, "->", clase)

# 5. Particion estratificada
print("\n[5/7] Dividiendo 70/15/15...")

indices = np.arange(len(df))

idx_train, idx_temp = train_test_split(
    indices,
    test_size=0.30,
    stratify=y,
    random_state=SEMILLA
)

idx_val, idx_test = train_test_split(
    idx_temp,
    test_size=0.50,
    stratify=y[idx_temp],
    random_state=SEMILLA
)

del idx_temp
gc.collect()

particiones = {
    "train": idx_train,
    "val": idx_val,
    "test": idx_test
}

for nombre, idx in particiones.items():
    print(nombre, ":", len(idx))

# Guardar etiquetas y trazabilidad
for nombre, idx in particiones.items():
    np.save(
        SALIDA / f"y_{nombre}.npy",
        y[idx].astype(np.int32)
    )

    df.iloc[idx][META + ["Label"]].to_csv(
        SALIDA / f"origen_{nombre}.csv",
        index=False
    )

# 6. Imputacion y escalado
print("\n[6/7] Ajustando transformadores...")

# Convertir predictores a float32 para reducir memoria
X = df[features].to_numpy(dtype=np.float32)

del df
gc.collect()

imputador = SimpleImputer(strategy="median")

# Solo entrenamiento
X_train = imputador.fit_transform(X[idx_train])

if X_train.shape[1] != len(features):
    raise ValueError(
        "La imputacion cambio el numero de variables"
    )

escalador = StandardScaler()

X_train = escalador.fit_transform(
    X_train
).astype(np.float32)

np.save(SALIDA / "X_train.npy", X_train)

del X_train
gc.collect()

# Aplicar transformadores sin reajustarlos
for nombre in ["val", "test"]:
    idx = particiones[nombre]

    X_parte = imputador.transform(X[idx])
    X_parte = escalador.transform(
        X_parte
    ).astype(np.float32)

    np.save(
        SALIDA / f"X_{nombre}.npy",
        X_parte
    )

    del X_parte
    gc.collect()

del X
gc.collect()

# Guardar transformadores
joblib.dump(
    imputador,
    SALIDA / "imputador.joblib"
)

joblib.dump(
    escalador,
    SALIDA / "escalador.joblib"
)

joblib.dump(
    encoder,
    SALIDA / "encoder_etiquetas.joblib"
)

configuracion = {
    "semilla": SEMILLA,
    "caracteristicas": features,
    "variables_eliminadas": eliminar,
    "variables_con_negativos_marcados": INVALIDAS,
    "clases": encoder.classes_.tolist(),
    "particion": {
        "entrenamiento": 0.70,
        "validacion": 0.15,
        "prueba": 0.15
    }
}

with open(
    SALIDA / "configuracion.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        configuracion,
        f,
        ensure_ascii=False,
        indent=2
    )

# 7. Verificaciones finales
print("\n[7/7] Verificando archivos...")

for nombre in ["train", "val", "test"]:
    datos = np.load(
        SALIDA / f"X_{nombre}.npy",
        mmap_mode="r"
    )

    etiquetas = np.load(
        SALIDA / f"y_{nombre}.npy"
    )

    print(
        f"{nombre}: X={datos.shape}, "
        f"y={etiquetas.shape}"
    )

    assert len(datos) == len(etiquetas)
    assert datos.shape[1] == 65

    conteos = np.bincount(
        etiquetas,
        minlength=len(encoder.classes_)
    )

    print("Distribucion por clase:")
    for clase, cantidad in zip(
        encoder.classes_, conteos
    ):
        print(f"  {clase}: {cantidad}")

print("\n=== PREPARACION FINALIZADA ===")
print("Resultados:", SALIDA)
