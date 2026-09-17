from pathlib import Path
from collections import deque
import argparse
import math
import time

import cv2
import numpy as np
from ultralytics import YOLO

from config.scenes import get_scene_config

from config.settings import (
    MODEL_FILENAME,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    TRACKER,
    GROUP_THRESHOLD,
    GROUP_PERSISTENCE,
    BOAT_PERSISTENCE,
    OUT_OF_HOURS_PERSISTENCE,
    RESTRICTED_ZONE_PERSISTENCE,
    REARM_PERSISTENCE,
    PRE_EVENT_SECONDS,
    POST_EVENT_SECONDS
)

from src.threat_logic import ThreatLogic
from src.event_logger import EventLogger
from src.actuator_controller import ActuatorController
from src.ui import draw_dashboard


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / MODEL_FILENAME
)


# ============================================================
# ARGUMENTOS DE TERMINAL
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "AgroGuard AI - "
        "Sistema de deteccion de amenazas"
    )
)

parser.add_argument(
    "--source",
    required=True,
    help=(
        "Fuente de video. "
        "Ejemplo: "
        "data/videos/camino/camino_01.mp4"
    )
)

args = parser.parse_args()


# ============================================================
# RESOLVER FUENTE
# ============================================================

source_argument = args.source


if source_argument.isdigit():

    VIDEO_SOURCE = int(
        source_argument
    )

    USING_CAMERA = True

else:

    VIDEO_SOURCE = (
        ROOT_DIR
        / source_argument
    ).resolve()

    USING_CAMERA = False


# ============================================================
# VERIFICAR MODELO
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"No se encontró el modelo:\n"
        f"{MODEL_PATH}"
    )


# ============================================================
# VERIFICAR FUENTE
# ============================================================

if not USING_CAMERA:

    if not VIDEO_SOURCE.exists():

        raise FileNotFoundError(
            f"No se encontró el video:\n"
            f"{VIDEO_SOURCE}"
        )


# ============================================================
# CONFIGURACIÓN DE ESCENA
# ============================================================

if USING_CAMERA:

    raise NotImplementedError(
        "La cámara en vivo todavía "
        "no tiene una configuración "
        "espacial definida."
    )


# ============================================================
# IDENTIFICAR VIDEO
# ============================================================

video_name = (
    VIDEO_SOURCE.name
)


# ============================================================
# OBTENER CONFIGURACIÓN ESPACIAL
# ============================================================

scene = get_scene_config(
    VIDEO_SOURCE
)

road_roi = scene.get(
    "road_roi"
)

road_line = scene.get(
    "road_line"
)

water_roi = scene.get(
    "water_roi"
)

restricted_side = scene.get(
    "restricted_side"
)

authorized_time = scene.get(
    "authorized_time",
    True
)


# ============================================================
# OBTENER FPS DEL VIDEO
# ============================================================

video_capture = cv2.VideoCapture(
    str(VIDEO_SOURCE)
)

VIDEO_FPS = video_capture.get(
    cv2.CAP_PROP_FPS
)

video_capture.release()


if VIDEO_FPS <= 0:

    raise RuntimeError(
        "No se pudieron determinar "
        "los FPS del video."
    )


print(
    f"Fuente: {VIDEO_SOURCE}"
)

print(
    f"Escena: {video_name}"
)

print(
    f"FPS: {VIDEO_FPS:.2f}"
)


# ============================================================
# BUFFER PRE-EVENTO
# ============================================================

PRE_EVENT_FRAMES = max(
    1,
    math.ceil(
        VIDEO_FPS
        * PRE_EVENT_SECONDS
    )
)


pre_event_buffer = deque(
    maxlen=PRE_EVENT_FRAMES
)


# ============================================================
# CLIPS PENDIENTES
# ============================================================

pending_clips = []


# ============================================================
# CARGAR MODELO
# ============================================================

print(
    f"Cargando modelo: "
    f"{MODEL_FILENAME}..."
)

model = YOLO(
    str(MODEL_PATH)
)

print(
    "Modelo cargado correctamente."
)


# ============================================================
# IDENTIFICAR CLASES
# ============================================================

person_id = None

boat_id = None


for class_id, class_name in (
    model.names.items()
):

    if class_name == "person":

        person_id = class_id

    elif class_name == "boat":

        boat_id = class_id


if person_id is None:

    raise RuntimeError(
        "El modelo no contiene "
        "la clase 'person'."
    )


if boat_id is None:

    raise RuntimeError(
        "El modelo no contiene "
        "la clase 'boat'."
    )


ACTIVE_CLASSES = [
    person_id,
    boat_id
]


print(
    f"Clases activas: "
    f"person={person_id}, "
    f"boat={boat_id}"
)


# ============================================================
# LÓGICA DE AMENAZAS
# ============================================================

threat_logic = ThreatLogic(
    group_threshold=(
        GROUP_THRESHOLD
    ),
    group_persistence=(
        GROUP_PERSISTENCE
    ),
    boat_persistence=(
        BOAT_PERSISTENCE
    ),
    out_of_hours_persistence=(
        OUT_OF_HOURS_PERSISTENCE
    ),
    restricted_zone_persistence=(
        RESTRICTED_ZONE_PERSISTENCE
    ),
    rearm_persistence=(
        REARM_PERSISTENCE
    )
)


# ============================================================
# REGISTRO DE EVENTOS
# ============================================================

event_logger = EventLogger(
    ROOT_DIR
)


# ============================================================
# CONTROL DE ACTUADORES
# ============================================================

actuator_controller = (
    ActuatorController()
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def point_inside_roi(
    point,
    roi
):

    if roi is None:

        return False


    polygon = np.array(
        roi,
        dtype=np.int32
    )


    result = cv2.pointPolygonTest(
        polygon,
        point,
        False
    )


    return result >= 0


def get_line_side(
    point,
    line
):

    if line is None:

        return 0


    x, y = point

    x1, y1 = line[0]
    x2, y2 = line[1]


    value = (
        (x2 - x1)
        * (y - y1)
        - (y2 - y1)
        * (x - x1)
    )


    if value > 0:

        return 1


    if value < 0:

        return -1


    return 0


# ============================================================
# CREAR EVENTO
# ============================================================

def create_event(
    event_type,
    frame,
    video_time,
    frame_number,
    person_count,
    boat_count
):

    event = event_logger.log_event(
        event_type=event_type,
        frame=frame,
        scene=video_name,
        video_time=video_time,
        frame_number=frame_number,
        person_count=person_count,
        boat_count=boat_count
    )


    clip = {

        "event_id": (
            event["event_id"]
        ),

        "event_type": (
            event_type
        ),

        "video_path": (
            Path(
                event["video_path"]
            )
        ),

        "end_time": (
            video_time
            + POST_EVENT_SECONDS
        ),

        "frames": [
            buffered_frame.copy()
            for buffered_frame
            in pre_event_buffer
        ]
    }


    pending_clips.append(
        clip
    )


    print(
        f"[EVENTO {event['event_id']}] "
        f"{event_type} | "
        f"Tiempo: {video_time:.2f} s | "
        f"Personas: {person_count} | "
        f"Boats: {boat_count}"
    )


# ============================================================
# FINALIZAR CLIPS COMPLETOS
# ============================================================

def finalize_completed_clips(
    current_time
):

    completed = []


    for clip in pending_clips:

        if current_time >= (
            clip["end_time"]
        ):

            event_logger.save_event_video(
                event_id=(
                    clip["event_id"]
                ),

                video_path=(
                    clip["video_path"]
                ),

                frames=(
                    clip["frames"]
                ),

                fps=VIDEO_FPS
            )


            completed.append(
                clip
            )


    for clip in completed:

        pending_clips.remove(
            clip
        )


# ============================================================
# INFERENCIA + TRACKING
# ============================================================

#results = model.track(
#    source=str(VIDEO_SOURCE),
#    conf=CONFIDENCE_THRESHOLD,
#    imgsz=IMAGE_SIZE,
#    classes=ACTIVE_CLASSES,
#    tracker=TRACKER,
#    persist=True,
#    stream=True,
#    verbose=False
#)

results = model.predict(
    source=str(VIDEO_SOURCE),
    conf=CONFIDENCE_THRESHOLD,
    imgsz=IMAGE_SIZE,
    classes=ACTIVE_CLASSES,
    stream=True,
    verbose=False
)


# ============================================================
# MEDICIÓN DE RENDIMIENTO
# ============================================================

processing_start = time.perf_counter()

processed_frames = 0


# ============================================================
# PROCESAMIENTO FRAME A FRAME
# ============================================================

for frame_number, result in enumerate(
    results
):

    processed_frames += 1

    # ========================================================
    # TIEMPO DEL VIDEO
    # ========================================================

    video_time = (
        frame_number
        / VIDEO_FPS
    )


    # ========================================================
    # FRAME ORIGINAL
    # ========================================================

    frame = (
        result.orig_img.copy()
    )


    # ========================================================
    # CONTADORES
    # ========================================================

    persons_in_road = 0

    persons_in_restricted_zone = 0

    boats_in_water = 0


    # ========================================================
    # ROAD ROI
    # ========================================================

    if road_roi is not None:

        road_polygon = np.array(
            road_roi,
            dtype=np.int32
        )


        cv2.polylines(
            frame,
            [road_polygon],
            True,
            (0, 255, 255),
            3
        )


    # ========================================================
    # WATER ROI
    # ========================================================

    if water_roi is not None:

        water_polygon = np.array(
            water_roi,
            dtype=np.int32
        )


        cv2.polylines(
            frame,
            [water_polygon],
            True,
            (255, 255, 0),
            3
        )


    # ========================================================
    # ROAD LINE
    # ========================================================

    if road_line is not None:

        cv2.line(
            frame,
            road_line[0],
            road_line[1],
            (255, 0, 255),
            3
        )


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
                box.xyxy[
                    0
                ].tolist()
            )


            # =================================================
            # PUNTOS DE REFERENCIA
            # =================================================

            center = (
                int(
                    (x1 + x2) / 2
                ),
                int(
                    (y1 + y2) / 2
                )
            )


            bottom_center = (
                int(
                    (x1 + x2) / 2
                ),
                y2
            )


            # =================================================
            # PERSONA
            # =================================================

            if class_name == "person":

                inside_road = (
                    point_inside_roi(
                        bottom_center,
                        road_roi
                    )
                )


                line_side = (
                    get_line_side(
                        bottom_center,
                        road_line
                    )
                )


                if inside_road:

                    persons_in_road += 1


                    if (
                        restricted_side
                        is not None
                        and line_side
                        == restricted_side
                    ):

                        persons_in_restricted_zone += 1

                        color = (
                            0,
                            0,
                            255
                        )

                        status = (
                            "RESTRICTED"
                        )


                    else:

                        color = (
                            0,
                            255,
                            0
                        )

                        status = (
                            "ROAD"
                        )


                else:

                    color = (
                        100,
                        100,
                        100
                    )

                    status = (
                        "IGNORED"
                    )


                reference_point = (
                    bottom_center
                )


                label = (
                    f"PERSON "
                    f"{confidence:.2f} "
                    f"[{status}]"
                )


            # =================================================
            # BOAT
            # =================================================

            elif class_name == "boat":

                inside_water = (
                    point_inside_roi(
                        center,
                        water_roi
                    )
                )


                if inside_water:

                    boats_in_water += 1

                    color = (
                        0,
                        255,
                        0
                    )

                    status = (
                        "WATER"
                    )


                else:

                    color = (
                        100,
                        100,
                        100
                    )

                    status = (
                        "IGNORED"
                    )


                reference_point = (
                    center
                )


                label = (
                    f"BOAT "
                    f"{confidence:.2f} "
                    f"[{status}]"
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
            # PUNTO DE REFERENCIA
            # =================================================

            cv2.circle(
                frame,
                reference_point,
                5,
                color,
                -1
            )


            # =================================================
            # ETIQUETA
            # =================================================

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        y1 - 8,
                        18
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )


    # ========================================================
    # REGLAS
    # ========================================================

    group_result = (
        threat_logic.evaluate_group(
            person_count=(
                persons_in_road
            ),
            timestamp=video_time
        )
    )


    boat_result = (
        threat_logic.evaluate_boat(
            boat_count=(
                boats_in_water
            ),
            timestamp=video_time
        )
    )


    out_of_hours_result = (
        threat_logic.evaluate_out_of_hours(
            person_count=(
                persons_in_road
            ),
            authorized_time=(
                authorized_time
            ),
            timestamp=video_time
        )
    )


    restricted_result = (
        threat_logic.evaluate_restricted_zone(
            person_count=(
                persons_in_restricted_zone
            ),
            timestamp=video_time
        )
    )


    # ========================================================
    # ESTADO GLOBAL DE AMENAZA
    # ========================================================

    any_threat_active = (
        group_result[
            "event_active"
        ]
        or boat_result[
            "event_active"
        ]
        or out_of_hours_result[
            "event_active"
        ]
        or restricted_result[
            "event_active"
        ]
    )


    # ========================================================
    # ACTUADORES
    # ========================================================

    actuator_controller.update(
        any_threat_active
    )


    actuator_state = (
        actuator_controller.get_state()
    )


    # ========================================================
    # INTERFAZ
    # ========================================================

    draw_dashboard(
        frame=frame,

        persons_in_road=(
            persons_in_road
        ),

        persons_in_restricted_zone=(
            persons_in_restricted_zone
        ),

        boats_in_water=(
            boats_in_water
        ),

        video_time=video_time,

        authorized_time=(
            authorized_time
        ),

        group_active=(
            group_result[
                "event_active"
            ]
        ),

        boat_active=(
            boat_result[
                "event_active"
            ]
        ),

        out_of_hours_active=(
            out_of_hours_result[
                "event_active"
            ]
        ),

        restricted_active=(
            restricted_result[
                "event_active"
            ]
        ),

        reflector_on=(
            actuator_state[
                "reflector"
            ]
        ),

        siren_on=(
            actuator_state[
                "siren"
            ]
        )
    )


    # ========================================================
    # BUFFER DE EVIDENCIA
    # ========================================================

    pre_event_buffer.append(
        frame.copy()
    )


    for clip in pending_clips:

        clip["frames"].append(
            frame.copy()
        )


    # ========================================================
    # EVENTOS NUEVOS
    # ========================================================

    if group_result["event"]:

        create_event(
            event_type=(
                "GROUP_DETECTED"
            ),
            frame=frame,
            video_time=video_time,
            frame_number=frame_number,
            person_count=(
                persons_in_road
            ),
            boat_count=(
                boats_in_water
            )
        )


    if boat_result["event"]:

        create_event(
            event_type=(
                "BOAT_INTRUSION"
            ),
            frame=frame,
            video_time=video_time,
            frame_number=frame_number,
            person_count=(
                persons_in_road
            ),
            boat_count=(
                boats_in_water
            )
        )


    if out_of_hours_result[
        "event"
    ]:

        create_event(
            event_type=(
                "OUT_OF_HOURS"
            ),
            frame=frame,
            video_time=video_time,
            frame_number=frame_number,
            person_count=(
                persons_in_road
            ),
            boat_count=(
                boats_in_water
            )
        )


    if restricted_result[
        "event"
    ]:

        create_event(
            event_type=(
                "RESTRICTED_ZONE_INTRUSION"
            ),
            frame=frame,
            video_time=video_time,
            frame_number=frame_number,
            person_count=(
                persons_in_restricted_zone
            ),
            boat_count=(
                boats_in_water
            )
        )


    # ========================================================
    # FINALIZAR CLIPS
    # ========================================================

    finalize_completed_clips(
        current_time=(
            video_time
        )
    )


    # ========================================================
    # MOSTRAR
    # ========================================================

    cv2.imshow(
        "AgroGuard AI",
        frame
    )


    if (
        cv2.waitKey(1)
        & 0xFF
        == ord("q")
    ):

        break


# ============================================================
# RESULTADOS DE RENDIMIENTO
# ============================================================

processing_end = time.perf_counter()

processing_time = (
    processing_end
    - processing_start
)


if processing_time > 0:

    processing_fps = (
        processed_frames
        / processing_time
    )

else:

    processing_fps = 0.0


real_time_factor = (
    processing_fps
    / VIDEO_FPS
    if VIDEO_FPS > 0
    else 0.0
)


print()
print(
    "========================================"
)

print(
    "RENDIMIENTO AGROGUARD"
)

print(
    "========================================"
)

print(
    f"Frames procesados: "
    f"{processed_frames}"
)

print(
    f"Tiempo real: "
    f"{processing_time:.2f} s"
)

print(
    f"FPS del video fuente: "
    f"{VIDEO_FPS:.2f}"
)

print(
    f"FPS reales de procesamiento: "
    f"{processing_fps:.2f}"
)

print(
    f"Factor de tiempo real: "
    f"{real_time_factor:.2f}x"
)


if processing_fps >= VIDEO_FPS:

    print(
        "Estado: TIEMPO REAL"
    )

else:

    print(
        "Estado: INFERIOR A TIEMPO REAL"
    )


print(
    "========================================"
)

print()


# ============================================================
# FINALIZAR CLIPS PENDIENTES
# ============================================================

for clip in pending_clips:

    event_logger.save_event_video(
        event_id=(
            clip["event_id"]
        ),
        video_path=(
            clip["video_path"]
        ),
        frames=(
            clip["frames"]
        ),
        fps=VIDEO_FPS
    )


pending_clips.clear()


# ============================================================
# APAGADO SEGURO
# ============================================================

actuator_controller.update(
    False
)

actuator_controller.cleanup()

cv2.destroyAllWindows()


print(
    "Procesamiento finalizado."
)