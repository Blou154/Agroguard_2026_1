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
# RUTAS / ARGUMENTOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / MODEL_FILENAME

parser = argparse.ArgumentParser(
    description=(
        "AgroGuard AI - prueba de baja latencia "
        "con descarte de frames atrasados"
    )
)

parser.add_argument(
    "--source",
    required=True
)

args = parser.parse_args()

VIDEO_SOURCE = (
    ROOT_DIR / args.source
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
# ESCENA
# ============================================================

video_name = VIDEO_SOURCE.name

scene = get_scene_config(
    VIDEO_SOURCE
)

road_roi = scene.get("road_roi")
road_line = scene.get("road_line")
water_roi = scene.get("water_roi")
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
        f"No se pudo abrir:\n"
        f"{VIDEO_SOURCE}"
    )


VIDEO_FPS = cap.get(
    cv2.CAP_PROP_FPS
)

TOTAL_FRAMES = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


if VIDEO_FPS <= 0:

    cap.release()

    raise RuntimeError(
        "FPS inválidos."
    )


SOURCE_DURATION = (
    TOTAL_FRAMES / VIDEO_FPS
    if TOTAL_FRAMES > 0
    else math.nan
)


print(f"Fuente: {VIDEO_SOURCE}")
print(f"Escena: {video_name}")
print(f"FPS fuente: {VIDEO_FPS:.2f}")
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

    name = str(
        class_name
    ).lower()

    if name == "person":
        person_id = int(class_id)

    elif name == "boat":
        boat_id = int(class_id)


if person_id is None:
    raise RuntimeError(
        "Falta clase person."
    )

if boat_id is None:
    raise RuntimeError(
        "Falta clase boat."
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
# LÓGICA / ACTUADORES / EVENTOS
# ============================================================

threat_logic = ThreatLogic(
    group_threshold=GROUP_THRESHOLD,
    group_persistence=GROUP_PERSISTENCE,
    boat_persistence=BOAT_PERSISTENCE,
    out_of_hours_persistence=OUT_OF_HOURS_PERSISTENCE,
    restricted_zone_persistence=RESTRICTED_ZONE_PERSISTENCE,
    rearm_persistence=REARM_PERSISTENCE
)

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
# FUNCIONES
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
        "event_id": event["event_id"],
        "event_type": event_type,
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
        f"[EVENTO {event['event_id']}] "
        f"{event_type} | "
        f"Tiempo video: "
        f"{video_time:.2f} s | "
        f"Personas: {person_count} | "
        f"Boats: {boat_count}"
    )


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
                event_id=clip["event_id"],
                video_path=clip[
                    "video_path"
                ],
                frames=clip["frames"],
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
# VARIABLES TEMPORALES
# ============================================================

runtime_start = (
    time.perf_counter()
)

processed_frames = 0
dropped_frames = 0
source_frame_number = -1


# Inicio REAL de una condición todavía no confirmada.
condition_start = {
    "group": None,
    "restricted": None,
    "out_of_hours": None,
    "boat": None
}


first_event_latency = {
    "group": None,
    "restricted": None,
    "out_of_hours": None,
    "boat": None
}


first_condition_for_gpio = None
first_gpio_runtime = None


# ============================================================
# PROCESAMIENTO LOW-LATENCY
# ============================================================

try:

    while True:

        ok, frame = cap.read()

        if not ok:
            break


        source_frame_number += 1
        processed_frames += 1


        runtime_time = (
            time.perf_counter()
            - runtime_start
        )


        video_time = (
            source_frame_number
            / VIDEO_FPS
        )


        # ====================================================
        # INFERENCIA
        # ====================================================

        result = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=IMAGE_SIZE,
            classes=ACTIVE_CLASSES,
            verbose=False
        )[0]


        persons_in_road = 0
        persons_in_restricted_zone = 0
        boats_in_water = 0


        # ====================================================
        # GEOMETRÍA
        # ====================================================

        if road_roi is not None:

            polygon = np.array(
                road_roi,
                dtype=np.int32
            )

            cv2.polylines(
                frame,
                [polygon],
                True,
                (0, 255, 255),
                3
            )


        if water_roi is not None:

            polygon = np.array(
                water_roi,
                dtype=np.int32
            )

            cv2.polylines(
                frame,
                [polygon],
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
                    model.names[class_id]
                ).lower()

                confidence = float(
                    box.conf[0]
                )

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )


                center = (
                    int((x1 + x2) / 2),
                    int((y1 + y2) / 2)
                )

                bottom_center = (
                    int((x1 + x2) / 2),
                    y2
                )


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
                            color = (0, 0, 255)
                            status = "RESTRICTED"

                        else:

                            color = (0, 255, 0)
                            status = "ROAD"

                    else:

                        color = (
                            100,
                            100,
                            100
                        )

                        status = "IGNORED"


                    reference_point = (
                        bottom_center
                    )

                    label = (
                        f"PERSON "
                        f"{confidence:.2f} "
                        f"[{status}]"
                    )


                elif class_name == "boat":

                    inside_water = (
                        point_inside_roi(
                            center,
                            water_roi
                        )
                    )


                    if inside_water:

                        boats_in_water += 1
                        color = (0, 255, 0)
                        status = "WATER"

                    else:

                        color = (
                            100,
                            100,
                            100
                        )

                        status = "IGNORED"


                    reference_point = center

                    label = (
                        f"BOAT "
                        f"{confidence:.2f} "
                        f"[{status}]"
                    )


                else:
                    continue


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
                        max(y1 - 8, 18)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA
                )


        # ====================================================
        # CONDICIONES CRUDAS
        #
        # Sirven para medir:
        # condición real detectada -> evento -> GPIO
        # ====================================================

        raw_condition = {
            "group": (
                persons_in_road
                > GROUP_THRESHOLD
            ),

            "restricted": (
                persons_in_restricted_zone
                > 0
            ),

            "out_of_hours": (
                persons_in_road > 0
                and not authorized_time
            ),

            "boat": (
                boats_in_water > 0
            )
        }


        for rule_name, active in (
            raw_condition.items()
        ):

            if active:

                if (
                    condition_start[
                        rule_name
                    ]
                    is None
                ):

                    condition_start[
                        rule_name
                    ] = runtime_time

            else:

                condition_start[
                    rule_name
                ] = None


        # ====================================================
        # THREAT LOGIC: RELOJ REAL
        # ====================================================

        group_result = (
            threat_logic.evaluate_group(
                person_count=persons_in_road,
                timestamp=runtime_time
            )
        )

        boat_result = (
            threat_logic.evaluate_boat(
                boat_count=boats_in_water,
                timestamp=runtime_time
            )
        )

        out_of_hours_result = (
            threat_logic.evaluate_out_of_hours(
                person_count=persons_in_road,
                authorized_time=authorized_time,
                timestamp=runtime_time
            )
        )

        restricted_result = (
            threat_logic.evaluate_restricted_zone(
                person_count=(
                    persons_in_restricted_zone
                ),
                timestamp=runtime_time
            )
        )


        rule_results = {
            "group": group_result,
            "restricted": restricted_result,
            "out_of_hours": out_of_hours_result,
            "boat": boat_result
        }


        # ====================================================
        # LATENCIA CONDICIÓN -> EVENTO
        # ====================================================

        for rule_name, result_rule in (
            rule_results.items()
        ):

            if (
                result_rule["event"]
                and first_event_latency[
                    rule_name
                ]
                is None
                and condition_start[
                    rule_name
                ]
                is not None
            ):

                latency = (
                    runtime_time
                    - condition_start[
                        rule_name
                    ]
                )

                first_event_latency[
                    rule_name
                ] = latency

                print(
                    f"[LATENCIA] "
                    f"{rule_name}: "
                    f"condición -> evento = "
                    f"{latency:.3f} s"
                )


        any_threat_active = any(
            r["event_active"]
            for r in rule_results.values()
        )


        # Identificar el inicio de la condición que dio
        # lugar a la primera amenaza global.
        if (
            any_threat_active
            and first_condition_for_gpio
            is None
        ):

            starts = [
                condition_start[name]
                for name, result_rule
                in rule_results.items()
                if (
                    result_rule[
                        "event_active"
                    ]
                    and condition_start[name]
                    is not None
                )
            ]

            if starts:

                first_condition_for_gpio = min(
                    starts
                )


        # ====================================================
        # GPIO
        # ====================================================

        actuator_controller.update(
            any_threat_active
        )

        actuator_state = (
            actuator_controller.get_state()
        )


        if (
            first_gpio_runtime is None
            and any_threat_active
            and (
                actuator_state["reflector"]
                or actuator_state["siren"]
            )
        ):

            first_gpio_runtime = (
                time.perf_counter()
                - runtime_start
            )


        # ====================================================
        # DASHBOARD
        # ====================================================

        draw_dashboard(
            frame=frame,
            persons_in_road=persons_in_road,
            persons_in_restricted_zone=(
                persons_in_restricted_zone
            ),
            boats_in_water=boats_in_water,
            video_time=video_time,
            authorized_time=authorized_time,
            group_active=group_result[
                "event_active"
            ],
            boat_active=boat_result[
                "event_active"
            ],
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
            reflector_on=actuator_state[
                "reflector"
            ],
            siren_on=actuator_state[
                "siren"
            ]
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
                "GROUP_DETECTED",
                frame,
                video_time,
                source_frame_number,
                persons_in_road,
                boats_in_water
            )


        if boat_result["event"]:

            create_event(
                "BOAT_INTRUSION",
                frame,
                video_time,
                source_frame_number,
                persons_in_road,
                boats_in_water
            )


        if out_of_hours_result["event"]:

            create_event(
                "OUT_OF_HOURS",
                frame,
                video_time,
                source_frame_number,
                persons_in_road,
                boats_in_water
            )


        if restricted_result["event"]:

            create_event(
                "RESTRICTED_ZONE_INTRUSION",
                frame,
                video_time,
                source_frame_number,
                persons_in_restricted_zone,
                boats_in_water
            )


        finalize_completed_clips(
            video_time
        )


        # ====================================================
        # MOSTRAR
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
                frame.shape[0] - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        cv2.imshow(
            "AgroGuard AI - Low Latency",
            frame
        )


        if (
            cv2.waitKey(1)
            & 0xFF
            == ord("q")
        ):

            break


        # ====================================================
        # DESCARTE DE FRAMES ATRASADOS
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


finally:

    # ========================================================
    # EVIDENCIAS PENDIENTES
    # ====================================================

    for clip in pending_clips:

        event_logger.save_event_video(
            event_id=clip["event_id"],
            video_path=clip[
                "video_path"
            ],
            frames=clip["frames"],
            fps=VIDEO_FPS
        )


    pending_clips.clear()


    # ========================================================
    # CIERRE GPIO
    # ====================================================

    try:

        actuator_controller.update(
            False
        )

        actuator_controller.cleanup()

    except Exception as exc:

        print(
            f"[WARN] Cleanup GPIO: "
            f"{exc}"
        )


    cap.release()

    cv2.destroyAllWindows()


# ============================================================
# RESULTADOS
# ============================================================

wall_elapsed = (
    time.perf_counter()
    - runtime_start
)


processing_fps = (
    processed_frames
    / wall_elapsed
    if wall_elapsed > 0
    else 0.0
)


total_seen = (
    processed_frames
    + dropped_frames
)


drop_percentage = (
    100.0
    * dropped_frames
    / total_seen
    if total_seen > 0
    else 0.0
)


print()
print("=" * 56)
print("MAIN2 - RESULTADOS DE BAJA LATENCIA")
print("=" * 56)

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
    f"{drop_percentage:.2f}%"
)

print(
    f"Duración fuente: "
    f"{SOURCE_DURATION:.2f} s"
)

print(
    f"Tiempo real: "
    f"{wall_elapsed:.2f} s"
)

print(
    f"FPS fuente: "
    f"{VIDEO_FPS:.2f}"
)

print(
    f"FPS de análisis: "
    f"{processing_fps:.2f}"
)


for rule_name, latency in (
    first_event_latency.items()
):

    if latency is not None:

        print(
            f"Latencia "
            f"{rule_name} "
            f"condición->evento: "
            f"{latency:.3f} s"
        )


if (
    first_condition_for_gpio
    is not None
    and first_gpio_runtime
    is not None
):

    total_latency = (
        first_gpio_runtime
        - first_condition_for_gpio
    )

    print(
        f"LATENCIA TOTAL "
        f"condición->GPIO: "
        f"{total_latency:.3f} s"
    )


print("=" * 56)
print(
    "Procesamiento finalizado."
)