from pathlib import Path
from collections import deque
from datetime import datetime
import argparse
import csv
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
from src.jetson_metrics import JetsonMetrics


# ============================================================
# RUTAS
# ============================================================

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / MODEL_FILENAME
)

METRICS_DIR = (
    ROOT_DIR
    / "runs"
    / "metrics"
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PERFORMANCE_CSV = (
    METRICS_DIR
    / "performance_runs.csv"
)

LATENCY_CSV = (
    METRICS_DIR
    / "latency_events.csv"
)


# ============================================================
# ARGUMENTOS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Sistema experimental - medición de "
        "procesamiento, recursos y latencia"
    )
)

parser.add_argument(
    "--source",
    required=True,
    help=(
        "Video de prueba. Ej.: "
        "data/validation/clips/camino/"
        "ROAD_GROUP_ZONE_002.mp4"
    )
)

args = parser.parse_args()


VIDEO_SOURCE = (
    ROOT_DIR
    / args.source
).resolve()


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"No se encontró el modelo:\n"
        f"{MODEL_PATH}"
    )


if not VIDEO_SOURCE.exists():

    raise FileNotFoundError(
        f"No se encontró el video:\n"
        f"{VIDEO_SOURCE}"
    )


# ============================================================
# IDENTIFICADOR DE EJECUCIÓN
# ============================================================

RUN_ID = datetime.now().strftime(
    "%Y%m%d_%H%M%S_%f"
)


def safe_filename(text):

    return "".join(
        character
        if (
            character.isalnum()
            or character in "-_"
        )
        else "_"
        for character in text
    )


VIDEO_SAFE_NAME = safe_filename(
    VIDEO_SOURCE.stem
)


TEGRSTATS_CSV = (
    METRICS_DIR
    / (
        f"tegrastats_"
        f"{RUN_ID}_"
        f"{VIDEO_SAFE_NAME}.csv"
    )
)


# ============================================================
# FUNCIONES DE CSV
# ============================================================

def append_csv_row(
    path,
    row
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = (
        path.exists()
        and path.stat().st_size > 0
    )

    with path.open(
        "a",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                row.keys()
            )
        )

        if not file_exists:

            writer.writeheader()

        writer.writerow(
            row
        )


def round_or_none(
    value,
    digits=4
):

    if value is None:
        return None

    return round(
        float(value),
        digits
    )


def format_metric(
    value,
    digits=2
):

    if value is None:
        return "N/A"

    return f"{value:.{digits}f}"


# ============================================================
# ESCENA
# ============================================================

video_name = (
    VIDEO_SOURCE.name
)

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
# CAPTURA
# ============================================================

cap = cv2.VideoCapture(
    str(VIDEO_SOURCE)
)


if not cap.isOpened():

    raise RuntimeError(
        f"No se pudo abrir el video:\n"
        f"{VIDEO_SOURCE}"
    )


VIDEO_FPS = cap.get(
    cv2.CAP_PROP_FPS
)

SOURCE_FRAME_COUNT = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


if VIDEO_FPS <= 0:

    cap.release()

    raise RuntimeError(
        "No se pudieron determinar "
        "los FPS del video."
    )


SOURCE_DURATION = (
    SOURCE_FRAME_COUNT
    / VIDEO_FPS
    if SOURCE_FRAME_COUNT > 0
    else math.nan
)


print(
    f"Fuente: {VIDEO_SOURCE}"
)

print(
    f"Escena: {video_name}"
)

print(
    f"FPS fuente: {VIDEO_FPS:.2f}"
)

print(
    f"Frames fuente: {SOURCE_FRAME_COUNT}"
)

if not math.isnan(
    SOURCE_DURATION
):

    print(
        f"Duración fuente: "
        f"{SOURCE_DURATION:.2f} s"
    )


# ============================================================
# MODELO
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


person_id = None
boat_id = None


for class_id, class_name in (
    model.names.items()
):

    class_name = str(
        class_name
    ).lower()

    if class_name == "person":

        person_id = int(
            class_id
        )

    elif class_name == "boat":

        boat_id = int(
            class_id
        )


if person_id is None:

    cap.release()

    raise RuntimeError(
        "El modelo no contiene "
        "la clase 'person'."
    )


if boat_id is None:

    cap.release()

    raise RuntimeError(
        "El modelo no contiene "
        "la clase 'boat'."
    )


ACTIVE_CLASSES = [
    person_id,
    boat_id
]


print(
    "Clases activas: "
    f"person={person_id}, "
    f"boat={boat_id}"
)


# ============================================================
# LÓGICA DE AMENAZAS
# ============================================================

threat_logic = ThreatLogic(
    group_threshold=GROUP_THRESHOLD,
    group_persistence=GROUP_PERSISTENCE,
    boat_persistence=BOAT_PERSISTENCE,
    out_of_hours_persistence=OUT_OF_HOURS_PERSISTENCE,
    restricted_zone_persistence=RESTRICTED_ZONE_PERSISTENCE,
    rearm_persistence=REARM_PERSISTENCE
)


# ============================================================
# EVENTOS / ACTUADORES
# ============================================================

event_logger = EventLogger(
    ROOT_DIR
)

actuator_controller = (
    ActuatorController()
)


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

pending_clips = []


# ============================================================
# FUNCIONES GEOMÉTRICAS
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

    return (
        cv2.pointPolygonTest(
            polygon,
            point,
            False
        )
        >= 0
    )


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
# CREACIÓN DE EVENTOS
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
        "event_id": event[
            "event_id"
        ],

        "event_type": (
            event_type
        ),

        "video_path": Path(
            event["video_path"]
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
        f"[EVENTO "
        f"{event['event_id']}] "
        f"{event_type} | "
        f"Tiempo video: "
        f"{video_time:.2f} s | "
        f"Personas: "
        f"{person_count} | "
        f"Boats: "
        f"{boat_count}"
    )

    return event


def finalize_completed_clips(
    current_video_time
):

    completed = []

    for clip in pending_clips:

        if (
            current_video_time
            >= clip["end_time"]
        ):

            event_logger.save_event_video(
                event_id=clip[
                    "event_id"
                ],

                video_path=clip[
                    "video_path"
                ],

                frames=clip[
                    "frames"
                ],

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
# MONITOR DE RECURSOS
# ============================================================

jetson_monitor = JetsonMetrics(
    interval_ms=500
)

print(
    "[METRICAS] Iniciando tegrastats..."
)

jetson_monitor.start()

print(
    "[METRICAS] Monitor de recursos activo."
)


# ============================================================
# VARIABLES DE MEDICIÓN
# ============================================================

runtime_start = (
    time.perf_counter()
)

processing_end = None


processed_frames = 0
dropped_frames = 0
source_frame_number = -1


condition_start = {
    "GROUP_DETECTED": None,
    "RESTRICTED_ZONE_INTRUSION": None,
    "BOAT_INTRUSION": None,
    "OUT_OF_HOURS": None
}


event_counter = {
    "GROUP_DETECTED": 0,
    "RESTRICTED_ZONE_INTRUSION": 0,
    "BOAT_INTRUSION": 0,
    "OUT_OF_HOURS": 0
}


latency_records = []


first_active_runtime = None
first_gpio_runtime = None


user_stopped = False


# ============================================================
# PROCESAMIENTO
#
# Este bloque conserva el principio de main2:
#
# - procesamiento con tiempo monotónico real;
# - descarte de frames atrasados;
# - evaluación de ROI;
# - ThreatLogic;
# - actuación física;
# - almacenamiento de evidencias.
#
# Añade:
#
# - t_condition
# - t_event
# - t_gpio
# - uso CPU
# - uso GPU
# - RAM
# - temperatura
# - CSV de desempeño
# ============================================================

try:

    while True:

        # -----------------------------------------------------
        # LEER SIGUIENTE FRAME A PROCESAR
        # -----------------------------------------------------

        ok, frame = cap.read()

        if not ok:
            break


        source_frame_number += 1
        processed_frames += 1


        video_time = (
            source_frame_number
            / VIDEO_FPS
        )


        # ====================================================
        # INFERENCIA
        # ====================================================

        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=IMAGE_SIZE,
            classes=ACTIVE_CLASSES,
            verbose=False
        )

        result = results[0]


        persons_in_road = 0
        persons_in_restricted_zone = 0
        boats_in_water = 0


        # ====================================================
        # DIBUJAR GEOMETRÍA
        # ====================================================

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


        if road_line is not None:

            cv2.line(
                frame,
                road_line[0],
                road_line[1],
                (255, 0, 255),
                3
            )


        # ====================================================
        # DETECCIONES
        # ====================================================

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                class_name = str(
                    model.names[
                        class_id
                    ]
                ).lower()

                confidence = float(
                    box.conf[0]
                )


                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[
                        0
                    ].tolist()
                )


                center = (
                    int(
                        (x1 + x2)
                        / 2
                    ),

                    int(
                        (y1 + y2)
                        / 2
                    )
                )


                bottom_center = (
                    int(
                        (x1 + x2)
                        / 2
                    ),

                    y2
                )


                # --------------------------------------------
                # PERSON
                # --------------------------------------------

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


                # --------------------------------------------
                # BOAT
                # --------------------------------------------

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


                # --------------------------------------------
                # DIBUJAR DETECCIÓN
                # --------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.circle(
                    frame,
                    reference_point,
                    5,
                    color,
                    -1
                )

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


        # ====================================================
        # INSTANTE REAL DE EVALUACIÓN DE LAS REGLAS
        #
        # Este es t_cond cuando una condición lógica aparece
        # por primera vez.
        # ====================================================

        logic_time = (
            time.perf_counter()
            - runtime_start
        )


        # ====================================================
        # CONDICIONES LÓGICAS INSTANTÁNEAS
        # ====================================================

        raw_condition = {

            "GROUP_DETECTED": (
                persons_in_road
                > GROUP_THRESHOLD
            ),

            "RESTRICTED_ZONE_INTRUSION": (
                persons_in_restricted_zone
                > 0
            ),

            "BOAT_INTRUSION": (
                boats_in_water
                > 0
            ),

            "OUT_OF_HOURS": (
                persons_in_road
                > 0
                and not authorized_time
            )
        }


        # ====================================================
        # REGISTRAR INICIO DE CADA CONDICIÓN
        # ====================================================

        for (
            event_type,
            active
        ) in raw_condition.items():

            if active:

                if (
                    condition_start[
                        event_type
                    ]
                    is None
                ):

                    condition_start[
                        event_type
                    ] = (
                        logic_time
                    )

            else:

                condition_start[
                    event_type
                ] = None


        # ====================================================
        # EVALUACIÓN DE THREAT LOGIC
        # ====================================================

        group_result = (
            threat_logic.evaluate_group(
                person_count=persons_in_road,
                timestamp=logic_time
            )
        )


        boat_result = (
            threat_logic.evaluate_boat(
                boat_count=boats_in_water,
                timestamp=logic_time
            )
        )


        out_of_hours_result = (
            threat_logic.evaluate_out_of_hours(
                person_count=persons_in_road,
                authorized_time=authorized_time,
                timestamp=logic_time
            )
        )


        restricted_result = (
            threat_logic.evaluate_restricted_zone(
                person_count=(
                    persons_in_restricted_zone
                ),
                timestamp=logic_time
            )
        )


        # ====================================================
        # EVENTOS CONFIRMADOS EN ESTE CICLO
        # ====================================================

        event_flags = {

            "GROUP_DETECTED": (
                group_result["event"]
            ),

            "RESTRICTED_ZONE_INTRUSION": (
                restricted_result["event"]
            ),

            "BOAT_INTRUSION": (
                boat_result["event"]
            ),

            "OUT_OF_HOURS": (
                out_of_hours_result["event"]
            )
        }


        event_confirm_times = {}


        for (
            event_type,
            confirmed
        ) in event_flags.items():

            if confirmed:

                event_confirm_times[
                    event_type
                ] = (
                    time.perf_counter()
                    - runtime_start
                )


        # ====================================================
        # ESTADO GLOBAL DE AMENAZA
        # ====================================================

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


        # ====================================================
        # ESTADO GPIO ANTES DE LA ORDEN
        # ====================================================

        actuator_state_before = (
            actuator_controller.get_state()
        )


        gpio_active_before = (

            actuator_state_before[
                "reflector"
            ]

            or actuator_state_before[
                "siren"
            ]
        )


        # ====================================================
        # ORDEN DE ACTUACIÓN
        # ====================================================

        actuator_controller.update(
            any_threat_active
        )


        # Momento inmediatamente posterior a que update()
        # terminó de ejecutar la orden sobre las salidas.
        gpio_command_time = (
            time.perf_counter()
            - runtime_start
        )


        actuator_state = (
            actuator_controller.get_state()
        )


        gpio_active_after = (

            actuator_state[
                "reflector"
            ]

            or actuator_state[
                "siren"
            ]
        )


        gpio_transition = (

            not gpio_active_before
            and gpio_active_after
        )


        # ====================================================
        # LATENCIA POR EVENTO CONFIRMADO
        # ====================================================

        for (
            event_type,
            t_event
        ) in event_confirm_times.items():

            t_condition = (
                condition_start.get(
                    event_type
                )
            )


            if t_condition is None:

                print(
                    "[WARN] "
                    f"No existe t_condition "
                    f"para {event_type}."
                )

                continue


            event_counter[
                event_type
            ] += 1


            t_gpio = (
                gpio_command_time
            )


            t_persistence = (
                t_event
                - t_condition
            )


            t_event_act = (
                t_gpio
                - t_event
            )


            t_condition_act = (
                t_gpio
                - t_condition
            )


            latency_record = {

                "run_id": RUN_ID,

                "video": video_name,

                "event_type": (
                    event_type
                ),

                "event_index": (
                    event_counter[
                        event_type
                    ]
                ),

                "source_frame_number": (
                    source_frame_number
                ),

                "video_time_s": round(
                    video_time,
                    6
                ),

                "persons_in_road": (
                    persons_in_road
                ),

                "persons_in_restricted": (
                    persons_in_restricted_zone
                ),

                "boats_in_water": (
                    boats_in_water
                ),

                "t_condition_s": round(
                    t_condition,
                    6
                ),

                "t_event_s": round(
                    t_event,
                    6
                ),

                "t_gpio_s": round(
                    t_gpio,
                    6
                ),

                "persistence_s": round(
                    t_persistence,
                    6
                ),

                "event_to_gpio_s": round(
                    t_event_act,
                    6
                ),

                "condition_to_gpio_s": round(
                    t_condition_act,
                    6
                ),

                "meets_2s": int(
                    t_condition_act
                    < 2.0
                ),

                "gpio_active_before": int(
                    gpio_active_before
                ),

                "gpio_active_after": int(
                    gpio_active_after
                ),

                "gpio_transition": int(
                    gpio_transition
                )
            }


            latency_records.append(
                latency_record
            )


            append_csv_row(
                LATENCY_CSV,
                latency_record
            )


            print(
                "[LATENCIA] "
                f"{event_type} | "
                f"cond->evento="
                f"{t_persistence:.3f} s | "
                f"evento->GPIO="
                f"{t_event_act * 1000:.2f} ms | "
                f"cond->GPIO="
                f"{t_condition_act:.3f} s | "
                f"<2s="
                f"{'SI' if t_condition_act < 2.0 else 'NO'}"
            )


            if (
                first_active_runtime
                is None
            ):

                first_active_runtime = (
                    t_event
                )


            if (
                first_gpio_runtime
                is None
                and gpio_active_after
            ):

                first_gpio_runtime = (
                    t_gpio
                )


        # ====================================================
        # UI
        # ====================================================

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

            video_time=(
                video_time
            ),

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


        # ====================================================
        # EVIDENCIA
        # ====================================================

        pre_event_buffer.append(
            frame.copy()
        )


        for clip in pending_clips:

            clip["frames"].append(
                frame.copy()
            )


        if group_result["event"]:

            create_event(
                event_type="GROUP_DETECTED",
                frame=frame,
                video_time=video_time,
                frame_number=source_frame_number,
                person_count=persons_in_road,
                boat_count=boats_in_water
            )


        if boat_result["event"]:

            create_event(
                event_type="BOAT_INTRUSION",
                frame=frame,
                video_time=video_time,
                frame_number=source_frame_number,
                person_count=persons_in_road,
                boat_count=boats_in_water
            )


        if out_of_hours_result["event"]:

            create_event(
                event_type="OUT_OF_HOURS",
                frame=frame,
                video_time=video_time,
                frame_number=source_frame_number,
                person_count=persons_in_road,
                boat_count=boats_in_water
            )


        if restricted_result["event"]:

            create_event(
                event_type=(
                    "RESTRICTED_ZONE_INTRUSION"
                ),
                frame=frame,
                video_time=video_time,
                frame_number=source_frame_number,
                person_count=(
                    persons_in_restricted_zone
                ),
                boat_count=boats_in_water
            )


        finalize_completed_clips(
            current_video_time=(
                video_time
            )
        )


        # ====================================================
        # MOSTRAR MÉTRICAS EN PANTALLA
        # ====================================================

        cv2.putText(
            frame,

            (
                f"Processed: "
                f"{processed_frames} | "
                f"Dropped: "
                f"{dropped_frames}"
            ),

            (
                20,
                frame.shape[0]
                - 20
            ),

            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        cv2.imshow(
            "Sistema - prueba instrumentada",
            frame
        )


        if (
            cv2.waitKey(1)
            & 0xFF
            == ord("q")
        ):

            user_stopped = True

            break


        # ====================================================
        # DESCARTAR FRAMES ATRASADOS
        #
        # Si el procesamiento tarda más que el periodo de la
        # fuente, se avanza sobre los frames antiguos sin
        # ejecutar YOLO sobre ellos.
        # ====================================================

        elapsed = (
            time.perf_counter()
            - runtime_start
        )


        target_frame = int(
            elapsed
            * VIDEO_FPS
        )


        while (
            source_frame_number + 1
            < target_frame
        ):

            grabbed = cap.grab()

            if not grabbed:
                break

            source_frame_number += 1
            dropped_frames += 1


except KeyboardInterrupt:

    user_stopped = True

    print(
        "\n[INFO] Ejecución interrumpida "
        "por el usuario."
    )


finally:

    # --------------------------------------------------------
    # FIN REAL DEL PROCESAMIENTO
    #
    # Se registra ANTES de guardar clips pendientes y hacer
    # cleanup, para que estas tareas no alteren los FPS.
    # --------------------------------------------------------

    processing_end = (
        time.perf_counter()
    )


    # ========================================================
    # DETENER TEGRSTATS
    # ========================================================

    try:

        jetson_monitor.stop()

        jetson_monitor.save_csv(
            TEGRSTATS_CSV
        )

        print(
            "[METRICAS] "
            f"Tegrastats guardado en:\n"
            f"{TEGRSTATS_CSV}"
        )

    except Exception as exc:

        print(
            "[WARN] Error al guardar "
            f"tegrastats: {exc}"
        )


    # ========================================================
    # CERRAR EVIDENCIAS PENDIENTES
    # ========================================================

    for clip in pending_clips:

        event_logger.save_event_video(
            event_id=clip[
                "event_id"
            ],

            video_path=clip[
                "video_path"
            ],

            frames=clip[
                "frames"
            ],

            fps=VIDEO_FPS
        )


    pending_clips.clear()


    # ========================================================
    # APAGADO SEGURO
    # ========================================================

    try:

        actuator_controller.update(
            False
        )

        actuator_controller.cleanup()

    except Exception as exc:

        print(
            "[WARN] Error durante "
            f"cleanup GPIO: {exc}"
        )


    cap.release()

    cv2.destroyAllWindows()


# ============================================================
# RESULTADOS DE PROCESAMIENTO
# ============================================================

wall_elapsed = (
    processing_end
    - runtime_start
)


processing_fps = (

    processed_frames
    / wall_elapsed

    if wall_elapsed > 0

    else 0.0
)


source_frames_seen = (
    processed_frames
    + dropped_frames
)


drop_percentage = (

    100.0
    * dropped_frames
    / source_frames_seen

    if source_frames_seen > 0

    else 0.0
)


realtime_ratio = (

    wall_elapsed
    / SOURCE_DURATION

    if (
        not math.isnan(
            SOURCE_DURATION
        )
        and SOURCE_DURATION > 0
    )

    else math.nan
)


# ============================================================
# RESUMEN DE RECURSOS
# ============================================================

resource_summary = (
    jetson_monitor.summary()
)


# ============================================================
# FILA DE DESEMPEÑO
# ============================================================

performance_row = {

    "run_id": RUN_ID,

    "video": video_name,

    "model": MODEL_FILENAME,

    "confidence_threshold": (
        CONFIDENCE_THRESHOLD
    ),

    "image_size": (
        IMAGE_SIZE
    ),

    "fps_source": round(
        VIDEO_FPS,
        4
    ),

    "source_frame_count": (
        SOURCE_FRAME_COUNT
    ),

    "source_duration_s": (
        round_or_none(
            SOURCE_DURATION,
            4
        )
    ),

    "wall_time_s": round(
        wall_elapsed,
        4
    ),

    "realtime_ratio_wall_over_source": (
        round_or_none(
            realtime_ratio,
            4
        )
    ),

    "source_frames_seen": (
        source_frames_seen
    ),

    "frames_processed": (
        processed_frames
    ),

    "frames_dropped": (
        dropped_frames
    ),

    "drop_percentage": round(
        drop_percentage,
        4
    ),

    "processing_fps": round(
        processing_fps,
        4
    ),

    "tegrastats_samples": (
        resource_summary[
            "sample_count"
        ]
    ),

    "tegrastats_interval_ms": (
        resource_summary[
            "interval_ms"
        ]
    ),

    "cpu_avg_pct": (
        round_or_none(
            resource_summary[
                "cpu"
            ]["avg"],
            4
        )
    ),

    "cpu_std_pct": (
        round_or_none(
            resource_summary[
                "cpu"
            ]["std"],
            4
        )
    ),

    "cpu_max_pct": (
        round_or_none(
            resource_summary[
                "cpu"
            ]["max"],
            4
        )
    ),

    "gpu_avg_pct": (
        round_or_none(
            resource_summary[
                "gpu"
            ]["avg"],
            4
        )
    ),

    "gpu_std_pct": (
        round_or_none(
            resource_summary[
                "gpu"
            ]["std"],
            4
        )
    ),

    "gpu_max_pct": (
        round_or_none(
            resource_summary[
                "gpu"
            ]["max"],
            4
        )
    ),

    "ram_avg_mb": (
        round_or_none(
            resource_summary[
                "ram"
            ]["avg"],
            2
        )
    ),

    "ram_std_mb": (
        round_or_none(
            resource_summary[
                "ram"
            ]["std"],
            2
        )
    ),

    "ram_max_mb": (
        round_or_none(
            resource_summary[
                "ram"
            ]["max"],
            2
        )
    ),

    "ram_total_mb": (
        round_or_none(
            resource_summary[
                "ram_total_mb"
            ],
            2
        )
    ),

    "temp_avg_c": (
        round_or_none(
            resource_summary[
                "temp"
            ]["avg"],
            3
        )
    ),

    "temp_std_c": (
        round_or_none(
            resource_summary[
                "temp"
            ]["std"],
            3
        )
    ),

    "temp_max_c": (
        round_or_none(
            resource_summary[
                "temp"
            ]["max"],
            3
        )
    ),

    "temp_peak_zone": (
        resource_summary[
            "temp_peak_zone"
        ]
    ),

    "events_measured": (
        len(
            latency_records
        )
    ),

    "user_stopped": int(
        user_stopped
    )
}


append_csv_row(
    PERFORMANCE_CSV,
    performance_row
)


# ============================================================
# IMPRESIÓN FINAL
# ============================================================

print()

print(
    "=" * 60
)

print(
    "RESULTADOS DE LA PRUEBA INSTRUMENTADA"
)

print(
    "=" * 60
)


print(
    f"Run ID: "
    f"{RUN_ID}"
)

print(
    f"Video: "
    f"{video_name}"
)

print(
    f"Frames fuente considerados: "
    f"{source_frames_seen}"
)

print(
    f"Frames procesados: "
    f"{processed_frames}"
)

print(
    f"Frames descartados: "
    f"{dropped_frames}"
)

print(
    f"Porcentaje descartado: "
    f"{drop_percentage:.2f} %"
)

print(
    f"Tiempo de procesamiento: "
    f"{wall_elapsed:.2f} s"
)

print(
    f"FPS fuente: "
    f"{VIDEO_FPS:.2f}"
)

print(
    f"FPS efectivos: "
    f"{processing_fps:.2f}"
)


if not math.isnan(
    realtime_ratio
):

    print(
        f"Relación tiempo real: "
        f"{realtime_ratio:.3f}"
    )


print()

print(
    "RECURSOS JETSON"
)

print(
    "-" * 60
)


print(
    "CPU promedio: "
    f"{format_metric(resource_summary['cpu']['avg'])} %"
)

print(
    "CPU máximo: "
    f"{format_metric(resource_summary['cpu']['max'])} %"
)

print(
    "GPU promedio: "
    f"{format_metric(resource_summary['gpu']['avg'])} %"
)

print(
    "GPU máximo: "
    f"{format_metric(resource_summary['gpu']['max'])} %"
)

print(
    "RAM promedio: "
    f"{format_metric(resource_summary['ram']['avg'])} MB"
)

print(
    "RAM máxima: "
    f"{format_metric(resource_summary['ram']['max'])} MB"
)

print(
    "Temperatura promedio: "
    f"{format_metric(resource_summary['temp']['avg'])} °C"
)

print(
    "Temperatura máxima: "
    f"{format_metric(resource_summary['temp']['max'])} °C"
)

print(
    "Zona del pico térmico: "
    f"{resource_summary['temp_peak_zone']}"
)

print(
    f"Muestras tegrastats: "
    f"{resource_summary['sample_count']}"
)


print()

print(
    "LATENCIA"
)

print(
    "-" * 60
)

print(
    f"Eventos medidos: "
    f"{len(latency_records)}"
)


if latency_records:

    condition_to_gpio_values = [
        row[
            "condition_to_gpio_s"
        ]
        for row in latency_records
    ]

    event_to_gpio_values = [
        row[
            "event_to_gpio_s"
        ]
        for row in latency_records
    ]

    mean_condition_to_gpio = (
        sum(
            condition_to_gpio_values
        )
        / len(
            condition_to_gpio_values
        )
    )

    max_condition_to_gpio = max(
        condition_to_gpio_values
    )

    mean_event_to_gpio = (
        sum(
            event_to_gpio_values
        )
        / len(
            event_to_gpio_values
        )
    )

    compliant_events = sum(
        1
        for value
        in condition_to_gpio_values
        if value < 2.0
    )


    print(
        "Promedio condición -> GPIO: "
        f"{mean_condition_to_gpio:.4f} s"
    )

    print(
        "Máximo condición -> GPIO: "
        f"{max_condition_to_gpio:.4f} s"
    )

    print(
        "Promedio evento -> GPIO: "
        f"{mean_event_to_gpio * 1000:.2f} ms"
    )

    print(
        "Cumplimiento < 2 s: "
        f"{compliant_events}/"
        f"{len(condition_to_gpio_values)}"
    )


if (
    first_active_runtime
    is not None
    and first_gpio_runtime
    is not None
):

    first_decision_gpio = (
        first_gpio_runtime
        - first_active_runtime
    )

    print(
        "Primera decisión -> GPIO: "
        f"{first_decision_gpio * 1000:.2f} ms"
    )


print()

print(
    "ARCHIVOS GENERADOS"
)

print(
    "-" * 60
)

print(
    f"Desempeño:\n"
    f"{PERFORMANCE_CSV}"
)

print(
    f"Latencias:\n"
    f"{LATENCY_CSV}"
)

print(
    f"Tegrastats:\n"
    f"{TEGRSTATS_CSV}"
)

print(
    "=" * 60
)

print(
    "Procesamiento finalizado."
)