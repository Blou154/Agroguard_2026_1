from pathlib import Path

import cv2


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

VIDEO_PATH = (
    ROOT_DIR
    / "data"
    / "validation"
    / "source"
    / "agua"
    / "vid_agu_001.mp4"
)


# ============================================================
# VARIABLES
# ============================================================

points = []


# ============================================================
# CALLBACK DEL MOUSE
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global points

    if event == cv2.EVENT_LBUTTONDOWN:

        points.append((x, y))

        print(f"Punto agregado: ({x}, {y})")


# ============================================================
# ABRIR VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

ret, frame = cap.read()

cap.release()


if not ret:

    raise RuntimeError(
        f"No se pudo abrir el video:\n{VIDEO_PATH}"
    )


# ============================================================
# VENTANA
# ============================================================

WINDOW_NAME = "AgroGuard - Selector de escena"

cv2.namedWindow(WINDOW_NAME)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


# ============================================================
# INTERFAZ
# ============================================================

while True:

    display = frame.copy()


    # --------------------------------------------------------
    # Dibujar puntos
    # --------------------------------------------------------

    for point in points:

        cv2.circle(
            display,
            point,
            6,
            (0, 0, 255),
            -1
        )


    # --------------------------------------------------------
    # Dibujar líneas
    # --------------------------------------------------------

    if len(points) >= 2:

        for i in range(len(points) - 1):

            cv2.line(
                display,
                points[i],
                points[i + 1],
                (0, 255, 255),
                2
            )


    # --------------------------------------------------------
    # Cerrar polígono visualmente
    # --------------------------------------------------------

    if len(points) >= 3:

        cv2.line(
            display,
            points[-1],
            points[0],
            (0, 255, 255),
            2
        )


    # --------------------------------------------------------
    # Instrucciones
    # --------------------------------------------------------

    cv2.putText(
        display,
        "Click: punto | R: reiniciar | S: mostrar coordenadas | Q: salir",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    cv2.imshow(
        WINDOW_NAME,
        display
    )


    key = cv2.waitKey(20) & 0xFF


    # --------------------------------------------------------
    # Reiniciar
    # --------------------------------------------------------

    if key == ord("r"):

        points = []

        print("\nSeleccion reiniciada.")


    # --------------------------------------------------------
    # Mostrar coordenadas
    # --------------------------------------------------------

    elif key == ord("s"):

        print("\nCoordenadas seleccionadas:")

        print(points)


    # --------------------------------------------------------
    # Salir
    # --------------------------------------------------------

    elif key == ord("q"):

        break


cv2.destroyAllWindows()