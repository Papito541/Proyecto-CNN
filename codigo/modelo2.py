from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, callbacks

# =====================================================
# 1. CONFIGURACIÓN
# =====================================================

BASE = Path(__file__).resolve().parent.parent

DATOS = BASE / "data" / "modelado_9_clases"

SALIDA = BASE / "resultados" / "cnn_9_clases"

SALIDA.mkdir(parents=True, exist_ok=True)

SEMILLA = 42
EPOCAS = 15
BATCH_SIZE = 512
LEARNING_RATE = 0.001

tf.keras.utils.set_random_seed(SEMILLA)

print("TensorFlow:", tf.__version__)

print(
    "Dispositivos GPU:",
    tf.config.list_physical_devices("GPU")
)

# =====================================================
# 2. CARGAR CONFIGURACIÓN
# =====================================================

with open(
    DATOS / "configuracion.json",
    encoding="utf-8"
) as f:

    config = json.load(f)

CLASES = config["clases"]

NUM_CLASES = len(CLASES)

NUM_FEATURES = len(config["caracteristicas"])

assert NUM_CLASES == 9
assert NUM_FEATURES == 65

print("\nClases:", CLASES)
print("Número de clases:", NUM_CLASES)
print("Características:", NUM_FEATURES)

# =====================================================
# 3. CARGAR DATOS
# =====================================================

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

# Adaptar dimensiones para Conv1D
X_train = X_train.reshape(
    -1, NUM_FEATURES, 1
)

X_val = X_val.reshape(
    -1, NUM_FEATURES, 1
)

print("\nEntrenamiento:", X_train.shape)
print("Validación:", X_val.shape)

# =====================================================
# 4. ARQUITECTURA CNN-1D
# =====================================================

modelo = models.Sequential([

    layers.Input(
        shape=(NUM_FEATURES, 1)
    ),

    layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling1D(
        pool_size=2
    ),

    layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),

    layers.GlobalAveragePooling1D(),

    layers.Dense(
        64,
        activation="relu"
    ),

    layers.Dropout(0.3),

    layers.Dense(
        NUM_CLASES,
        activation="softmax"
    )

], name="CNN_1D_9_Clases")

# =====================================================
# 5. COMPILACIÓN
# =====================================================

modelo.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

modelo.summary()

# Guardar arquitectura en TXT
with open(
    SALIDA / "arquitectura.txt",
    "w",
    encoding="utf-8"
) as f:

    modelo.summary(
        print_fn=lambda linea: f.write(linea + "\n")
    )

# =====================================================
# 6. CALLBACKS
# =====================================================

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

# =====================================================
# 7. ENTRENAMIENTO
# =====================================================

print("\n=== INICIANDO ENTRENAMIENTO CNN-1D ===")

historial = modelo.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCAS,

    batch_size=BATCH_SIZE,

    callbacks=controles,

    verbose=1

)

# =====================================================
# 8. GUARDAR RESUMEN
# =====================================================

indice_mejor = int(
    np.argmin(historial.history["val_loss"])
)

resumen = {

    "arquitectura": "CNN-1D",

    "numero_clases": NUM_CLASES,

    "caracteristicas": NUM_FEATURES,

    "parametros": int(modelo.count_params()),

    "semilla": SEMILLA,

    "batch_size": BATCH_SIZE,

    "learning_rate_inicial": LEARNING_RATE,

    "epocas_ejecutadas": len(
        historial.history["loss"]
    ),

    "mejor_epoca": indice_mejor + 1,

    "mejor_val_loss": float(
        historial.history["val_loss"][indice_mejor]
    ),

    "accuracy_en_mejor_epoca": float(
        historial.history["val_accuracy"][indice_mejor]
    ),

    "registros_train": len(y_train),

    "registros_val": len(y_val)

}

with open(
    SALIDA / "resumen_entrenamiento.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        resumen,
        f,
        indent=4
    )

print("\n=== ENTRENAMIENTO FINALIZADO ===")

print(
    json.dumps(
        resumen,
        indent=4
    )
)

print("\nModelo guardado:", ruta_modelo)