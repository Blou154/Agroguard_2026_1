from pathlib import Path
import argparse
import time

import cv2
from ultralytics import YOLO


# ============================================================
# RUTAS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "agroguard_yolov8n_v1.pt"
)


# ============================================================
# ARGUMENTOS
# ============================================================

parser = argparse.ArgumentParser(
    description="Prueba de cámara en vivo - AgroGuard AI"
)

parser.add_argument(
    "--camera",
    type=int,
    default=0,
    help="Índice de la cámara. Normalmente 0."
)

parser.add_argument(
    "--conf",
    type=float,
    default=0.25,
    help="Umbral de confianza."
)

args = parser.parse_args()


# ============================================================
# VERIFICAR MODELO
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"No se encontró el modelo:\n{MODEL_PATH}"
    )


# ============================================================
# CARGAR YOLO
# ============================================================

print(
    f"Cargando modelo:\n{MODEL_PATH}"
)

model = YOLO(
    str(MODEL_PATH)
)


# ============================================================
# IDENTIFICAR CLASES POR NOMBRE
# ============================================================

person_id = None
boat_id = None


for class_id, class_name in model.names.items():

    if class_name == "person":
        person_id = class_id

    elif class_name == "boat":
        boat_id = class_id


if person_id is None:
    raise RuntimeError(
        "El modelo no contiene la clase person."
    )

if boat_id is None:
    raise RuntimeError(
        "El modelo no contiene la clase boat."
    )


ACTIVE_CLASSES = [
    person_id,
    boat_id
]


print(
    f"Clases del modelo: {model.names}"
)

print(
    f"person = {person_id}"
)

print(
    f"boat = {boat_id}"
)


# ============================================================
# ABRIR CÁMARA
# ============================================================

camera = cv2.VideoCapture(
    args.camera,
    cv2.CAP_DSHOW
)


if not camera.isOpened():

    raise RuntimeError(
        f"No se pudo abrir la cámara {args.camera}."
    )


# Intentar resolución HD
camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


width = int(
    camera.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    camera.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

fps_camera = camera.get(
    cv2.CAP_PROP_FPS
)


print()
print(
    "========================================"
)

print(
    "AGROGUARD - CÁMARA EN VIVO"
)

print(
    "========================================"
)

print(
    f"Cámara: {args.camera}"
)

print(
    f"Resolución: {width}x{height}"
)

print(
    f"FPS reportados: {fps_camera:.2f}"
)

print(
    f"Confidence: {args.conf}"
)

print(
    "Presiona Q para salir."
)

print(
    "========================================"
)

print()


# ============================================================
# VARIABLES DE RENDIMIENTO
# ============================================================

previous_time = time.monotonic()

fps_display = 0.0


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

while True:

    success, frame = camera.read()


    if not success:

        print(
            "No se pudo leer un frame."
        )

        break


    # ========================================================
    # INFERENCIA
    # ========================================================

    results = model.predict(
        source=frame,
        conf=args.conf,
        imgsz=640,
        classes=ACTIVE_CLASSES,
        verbose=False
    )


    result = results[0]


    persons = 0
    boats = 0


    # ========================================================
    # DETECCIONES
    # ========================================================

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            class_name = (
                model.names[
                    class_id
                ]
            )

            confidence = float(
                box.conf[0]
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )


            if class_name == "person":

                persons += 1

                color = (
                    0,
                    255,
                    0
                )


            elif class_name == "boat":

                boats += 1

                color = (
                    255,
                    255,
                    0
                )


            else:

                continue


            # =================================================
            # BOUNDING BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )


            # =================================================
            # ETIQUETA
            # =================================================

            label = (
                f"{class_name.upper()} "
                f"{confidence:.2f}"
            )


            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        y1 - 8,
                        20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )


    # ========================================================
    # FPS DE PROCESAMIENTO
    # ========================================================

    current_time = (
        time.monotonic()
    )

    elapsed = (
        current_time
        - previous_time
    )

    previous_time = (
        current_time
    )


    if elapsed > 0:

        current_fps = (
            1.0
            / elapsed
        )

        fps_display = (
            0.9
            * fps_display
            + 0.1
            * current_fps
        )


    # ========================================================
    # PANEL SIMPLE
    # ========================================================

    cv2.putText(
        frame,
        f"PERSONS: {persons}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"BOATS: {boats}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"FPS: {fps_display:.1f}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # MOSTRAR
    # ========================================================

    cv2.imshow(
        "AgroGuard AI - Camera Test",
        frame
    )


    key = (
        cv2.waitKey(1)
        & 0xFF
    )


    if key == ord("q"):

        break


# ============================================================
# CIERRE
# ============================================================

camera.release()

cv2.destroyAllWindows()

print(
    "Prueba de cámara finalizada."
)