
from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, callbacks

BASE = Path(__file__).resolve().parent.parent
DATOS = BASE / "data" / "modelado"
SALIDA = BASE / "resultados" / "cnn"
SALIDA.mkdir(parents=True, exist_ok=True)

SEMILLA = 42
EPOCAS = 15
BATCH_SIZE = 512

tf.keras.utils.set_random_seed(SEMILLA)

print("TensorFlow:", tf.__version__)
print("Dispositivos GPU:", tf.config.list_physical_devices("GPU"))

# 1. Cargar entrenamiento y validación
X_train = np.load(
    DATOS / "X_train.npy",
    mmap_mode="r"
)

y_train = np.load(
    DATOS / "y_train.npy",
    mmap_mode="r"
)

X_val = np.load(
    DATOS / "X_val.npy",
    mmap_mode="r"
)

y_val = np.load(
    DATOS / "y_val.npy",
    mmap_mode="r"
)

# Conv1D requiere: muestras, pasos, canales
X_train = X_train.reshape(-1, 65, 1)
X_val = X_val.reshape(-1, 65, 1)

print("Entrenamiento:", X_train.shape)
print("Validacion:", X_val.shape)

with open(
    DATOS / "configuracion.json",
    encoding="utf-8"
) as f:
    config = json.load(f)

NUM_CLASES = len(config["clases"])

# 2. Arquitectura CNN
modelo = models.Sequential([
    layers.Input(shape=(65, 1)),

    layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),

    layers.GlobalAveragePooling1D(),

    layers.Dense(64, activation="relu"),

    layers.Dropout(0.3),

    layers.Dense(
        NUM_CLASES,
        activation="softmax"
    )
])

# 3. Compilación
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

modelo.summary()

# 4. Callbacks
ruta_modelo = SALIDA / "mejor_modelo.keras"

controles = [
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    callbacks.ModelCheckpoint(
        filepath=str(ruta_modelo),
        monitor="val_loss",
        save_best_only=True
    ),

    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001
    ),

    callbacks.CSVLogger(
        str(SALIDA / "historial.csv")
    )
]

# 5. Entrenamiento
print("\n=== INICIANDO ENTRENAMIENTO ===")

historial = modelo.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCAS,
    batch_size=BATCH_SIZE,
    callbacks=controles,
    verbose=1
)

print("\n=== ENTRENAMIENTO FINALIZADO ===")

print(
    "Mejor val_loss:",
    min(historial.history["val_loss"])
)

print(
    "Epocas ejecutadas:",
    len(historial.history["loss"])
)

print("Modelo guardado:", ruta_modelo)
