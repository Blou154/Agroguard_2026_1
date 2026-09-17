from pathlib import Path
import csv
import math
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import torch
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
)

from src.threat_logic import ThreatLogic


# ============================================================
# RUTAS
# ============================================================

MODEL_PATH = ROOT_DIR / "models" / MODEL_FILENAME

PLAN_PATH = (
    ROOT_DIR
    / "data"
    / "validation"
    / "validation_cut_plan.csv"
)

CLIPS_DIR = (
    ROOT_DIR
    / "data"
    / "validation"
    / "clips"
)

# Se usan nombres nuevos para no sobrescribir la campaña anterior.
RESULTS_PATH = (
    ROOT_DIR
    / "data"
    / "validation"
    / "validation_results_main_metrics_tegrastats_warmup.csv"
)

SUMMARY_PATH = (
    ROOT_DIR
    / "data"
    / "validation"
    / "validation_summary_main_metrics_tegrastats_warmup.csv"
)

PERFORMANCE_SUMMARY_PATH = (
    ROOT_DIR
    / "data"
    / "validation"
    / "validation_performance_summary_tegrastats_warmup.csv"
)

RESOURCES_RAW_PATH = (
    ROOT_DIR
    / "data"
    / "validation"
    / "validation_resources_raw_tegrastats_warmup.csv"
)

RESOURCES_SUMMARY_PATH = (
    ROOT_DIR
    / "data"
    / "validation"
    / "validation_resources_summary_tegrastats_warmup.csv"
)

# Intervalo de muestreo de tegrastats.
# 500 ms -> aproximadamente 2 muestras/s.
TEGRASTATS_INTERVAL_MS = 500

# Inferencias de calentamiento ejecutadas ANTES de iniciar tegrastats y
# ANTES de medir cualquier clip. Su objetivo es absorber el coste de
# inicialización de CUDA, carga perezosa de kernels y preparación inicial
# del predictor para que la campaña mida el régimen estable del sistema.
WARMUP_ITERATIONS = 5


# ============================================================
# UTILIDADES
# ============================================================

def parse_expected(value):
    if value is None:
        return None

    text = str(value).strip().upper()

    if text in {"N/A", "NA", "", "NONE"}:
        return None

    if text in {"1", "TRUE", "YES", "SI", "SÍ"}:
        return True

    if text in {"0", "FALSE", "NO"}:
        return False

    raise ValueError(f"Valor no reconocido: {value!r}")


def bool_to_csv(value):
    if value is None:
        return "N/A"

    return 1 if value else 0


def point_inside_roi(point, roi):
    if roi is None:
        return False

    polygon = np.array(roi, dtype=np.int32)

    return (
        cv2.pointPolygonTest(
            polygon,
            point,
            False,
        )
        >= 0
    )


def get_line_side(point, line):
    if line is None:
        return 0

    x, y = point
    x1, y1 = line[0]
    x2, y2 = line[1]

    value = (
        (x2 - x1) * (y - y1)
        - (y2 - y1) * (x - x1)
    )

    if value > 0:
        return 1

    if value < 0:
        return -1

    return 0


def evaluate_pass(expected, detected):
    if expected is None:
        return None

    return expected == detected


def csv_float(value):
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    return f"{value:.4f}"


def mean(values):
    if not values:
        return math.nan

    return sum(values) / len(values)


def sample_std(values):
    if len(values) < 2:
        return math.nan

    avg = mean(values)

    return math.sqrt(
        sum((value - avg) ** 2 for value in values)
        / (len(values) - 1)
    )


def stats(values):
    """Devuelve n, media, desviación muestral, mínimo y máximo."""
    clean = [
        float(value)
        for value in values
        if value is not None
        and not (
            isinstance(value, float)
            and math.isnan(value)
        )
    ]

    if not clean:
        return {
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "max": math.nan,
        }

    return {
        "n": len(clean),
        "mean": mean(clean),
        "std": sample_std(clean),
        "min": min(clean),
        "max": max(clean),
    }


# ============================================================
# MONITOR DE RECURSOS NVIDIA TEGRASTATS
# ============================================================

class TegraStatsMonitor:
    """
    Ejecuta tegrastats durante toda la campaña y almacena muestras con
    timestamp monotónico para asociarlas a cada clip.

    Variables extraídas:
      - CPU global (%): promedio de todos los núcleos reportados.
        Los núcleos "off" se contabilizan como 0 %.
      - GPU (%): GR3D_FREQ.
      - RAM usada y total (MB).
      - Temperatura máxima entre las zonas térmicas reportadas (°C).
      - Temperatura CPU y GPU cuando esas zonas aparecen explícitamente.
    """

    RAM_RE = re.compile(
        r"\bRAM\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)MB"
    )

    CPU_RE = re.compile(
        r"\bCPU\s+\[([^\]]+)\]"
    )

    GPU_RE = re.compile(
        r"\bGR3D_FREQ\s+(\d+(?:\.\d+)?)%"
    )

    TEMP_RE = re.compile(
        r"([A-Za-z0-9_]+)@(-?\d+(?:\.\d+)?)C"
    )

    PERCENT_RE = re.compile(
        r"(\d+(?:\.\d+)?)%"
    )

    def __init__(self, interval_ms=500):
        self.interval_ms = int(interval_ms)
        self.executable = self._find_executable()

        self.samples = []
        self.lock = threading.Lock()

        self.process = None
        self.thread = None
        self.enabled = False
        self.monitor_start_perf = None

    @staticmethod
    def _find_executable():
        candidates = [
            shutil.which("tegrastats"),
            "/usr/bin/tegrastats",
        ]

        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return str(candidate)

        return None

    @classmethod
    def parse_line(cls, line):
        ram_used_mb = None
        ram_total_mb = None
        cpu_util_pct = None
        gpu_util_pct = None
        temperature_max_c = None
        cpu_temp_c = None
        gpu_temp_c = None

        ram_match = cls.RAM_RE.search(line)

        if ram_match:
            ram_used_mb = float(ram_match.group(1))
            ram_total_mb = float(ram_match.group(2))

        cpu_match = cls.CPU_RE.search(line)

        if cpu_match:
            core_values = []

            for item in cpu_match.group(1).split(","):
                item = item.strip()

                if not item:
                    continue

                if "off" in item.lower():
                    core_values.append(0.0)
                    continue

                pct_match = cls.PERCENT_RE.search(item)

                if pct_match:
                    core_values.append(
                        float(pct_match.group(1))
                    )

            if core_values:
                cpu_util_pct = mean(core_values)

        gpu_match = cls.GPU_RE.search(line)

        if gpu_match:
            gpu_util_pct = float(gpu_match.group(1))

        temperatures = {
            name.lower(): float(value)
            for name, value in cls.TEMP_RE.findall(line)
        }

        if temperatures:
            temperature_max_c = max(temperatures.values())
            cpu_temp_c = temperatures.get("cpu")
            gpu_temp_c = temperatures.get("gpu")

        return {
            "cpu_util_pct": cpu_util_pct,
            "gpu_util_pct": gpu_util_pct,
            "ram_used_mb": ram_used_mb,
            "ram_total_mb": ram_total_mb,
            "temperature_max_c": temperature_max_c,
            "cpu_temp_c": cpu_temp_c,
            "gpu_temp_c": gpu_temp_c,
        }

    def start(self):
        if self.enabled:
            return True

        if self.executable is None:
            print(
                "ADVERTENCIA: no se encontró 'tegrastats'. "
                "Las métricas CPU/GPU/RAM/temperatura quedarán vacías."
            )
            return False

        command = [
            self.executable,
            "--interval",
            str(self.interval_ms),
        ]

        try:
            self.monitor_start_perf = time.perf_counter()

            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            print(
                "ADVERTENCIA: no se pudo iniciar tegrastats: "
                f"{exc}"
            )
            return False

        self.enabled = True

        self.thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
        )

        self.thread.start()

        print(
            f"tegrastats activo: {self.executable} "
            f"(intervalo={self.interval_ms} ms)"
        )

        return True

    def _read_loop(self):
        if self.process is None or self.process.stdout is None:
            return

        for raw_line in self.process.stdout:
            if not self.enabled:
                break

            line = raw_line.strip()

            if not line:
                continue

            now_perf = time.perf_counter()
            parsed = self.parse_line(line)

            parsed["_perf_counter"] = now_perf
            parsed["timestamp_s"] = (
                now_perf - self.monitor_start_perf
                if self.monitor_start_perf is not None
                else math.nan
            )
            parsed["raw_line"] = line

            with self.lock:
                self.samples.append(parsed)

    def stop(self):
        if not self.enabled and self.process is None:
            return

        self.enabled = False

        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)
            except OSError:
                pass

        if (
            self.thread is not None
            and self.thread.is_alive()
        ):
            self.thread.join(timeout=2.0)

        self.process = None

    def get_samples(self):
        with self.lock:
            return [dict(sample) for sample in self.samples]

    def samples_between(self, start_perf, end_perf):
        with self.lock:
            return [
                dict(sample)
                for sample in self.samples
                if start_perf
                <= sample["_perf_counter"]
                <= end_perf
            ]

    def summarize_window(self, start_perf, end_perf):
        window = self.samples_between(
            start_perf,
            end_perf,
        )

        metrics = {
            "cpu_util_pct": [
                item["cpu_util_pct"]
                for item in window
            ],
            "gpu_util_pct": [
                item["gpu_util_pct"]
                for item in window
            ],
            "ram_used_mb": [
                item["ram_used_mb"]
                for item in window
            ],
            "temperature_c": [
                item["temperature_max_c"]
                for item in window
            ],
        }

        summary = {
            "resource_samples": len(window),
        }

        for metric_name, values in metrics.items():
            metric_stats = stats(values)

            for stat_name in (
                "mean",
                "std",
                "min",
                "max",
            ):
                summary[
                    f"{metric_name}_{stat_name}"
                ] = metric_stats[stat_name]

        return summary


# ============================================================
# ADQUISICIÓN DE ÚLTIMO FOTOGRAMA
# ============================================================

class LatestFrameReader:
    """
    Reproduce un archivo de video a su FPS nominal y conserva únicamente
    el fotograma más reciente, emulando el comportamiento de una fuente
    de video en tiempo real.

    Cada fotograma capturado recibe un ID secuencial. El hilo principal
    usa esos IDs para contabilizar explícitamente los fotogramas que fueron
    reemplazados antes de llegar a inferencia.
    """

    def __init__(self, video_path):
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir el video: {self.video_path}"
            )

        self.fps = float(
            self.cap.get(cv2.CAP_PROP_FPS)
        )

        self.source_frame_count = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if self.fps <= 0:
            self.cap.release()
            raise RuntimeError(
                f"FPS inválido en: {self.video_path}"
            )

        self.source_duration_s = (
            self.source_frame_count / self.fps
            if self.source_frame_count > 0
            else math.nan
        )

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_id = -1
        self.frames_captured = 0

        self.started = False
        self.ended = False
        self.stop_requested = False
        self.thread = None
        self.start_time = None

    def start(self):
        if self.started:
            return

        self.started = True
        self.start_time = time.perf_counter()

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
        )

        self.thread.start()

    def _capture_loop(self):
        frame_period = 1.0 / self.fps
        frame_id = 0

        while not self.stop_requested:
            target_time = (
                self.start_time
                + frame_id * frame_period
            )

            delay = target_time - time.perf_counter()

            if delay > 0:
                time.sleep(delay)

            ok, frame = self.cap.read()

            if not ok:
                break

            with self.lock:
                self.latest_frame = frame
                self.latest_id = frame_id
                self.frames_captured += 1

            frame_id += 1

        with self.lock:
            self.ended = True

        self.cap.release()

    def get_latest(self):
        with self.lock:
            if self.latest_frame is None:
                frame = None
            else:
                frame = self.latest_frame.copy()

            return (
                self.latest_id,
                frame,
                self.ended,
                self.frames_captured,
            )

    def stop(self):
        self.stop_requested = True

        if (
            self.thread is not None
            and self.thread.is_alive()
        ):
            self.thread.join(timeout=2.0)

        if self.cap.isOpened():
            self.cap.release()


# ============================================================
# CALENTAMIENTO PREVIO DEL MODELO
# ============================================================

def warmup_model(model, active_classes, iterations=5):
    """
    Ejecuta inferencias sintéticas antes de comenzar la medición.

    Estas iteraciones NO se incluyen en tegrastats, FPS, latencias ni en
    ningún CSV de validación. El objetivo es retirar del intervalo medido
    el transitorio de arranque asociado a la primera utilización del
    backend de inferencia y de CUDA.

    Después del calentamiento se elimina únicamente el objeto predictor
    de Ultralytics para conservar el mismo estado de inicio por clip que
    utiliza la campaña. La inicialización de CUDA/kernels ya realizada
    permanece en el proceso.
    """
    iterations = max(1, int(iterations))

    # Imagen compatible con el tamaño de entrada empleado por la campaña.
    # Ultralytics realizará el mismo preprocesamiento usado en model.predict.
    warmup_frame = np.zeros(
        (int(IMAGE_SIZE), int(IMAGE_SIZE), 3),
        dtype=np.uint8,
    )

    print()
    print("=" * 78)
    print(
        "WARM-UP DEL MODELO "
        f"({iterations} inferencias, fuera de la medición)"
    )
    print("=" * 78)

    warmup_times_ms = []

    for iteration in range(1, iterations + 1):
        start = time.perf_counter()

        _ = model.predict(
            source=warmup_frame,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=IMAGE_SIZE,
            classes=active_classes,
            verbose=False,
        )[0]

        # CUDA es asíncrono. La sincronización garantiza que el tiempo de
        # calentamiento cubra realmente el trabajo lanzado a la GPU antes
        # de comenzar la siguiente iteración o la campaña medida.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        warmup_times_ms.append(elapsed_ms)

        print(
            f"Warm-up {iteration:02d}/{iterations}: "
            f"{elapsed_ms:.3f} ms"
        )

    # La campaña original reinicia predictor antes de cada clip. Se mantiene
    # esa condición para no cambiar el procedimiento entre segmentos.
    model.predictor = None

    print(
        "Warm-up finalizado. "
        "A partir de este punto comienzan las mediciones."
    )
    print("=" * 78)

    return warmup_times_ms


# ============================================================
# CARGAR PLAN / MODELO
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró:\n{MODEL_PATH}"
    )

if not PLAN_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró:\n{PLAN_PATH}"
    )

with PLAN_PATH.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as file:
    plan_rows = list(csv.DictReader(file))


model = YOLO(str(MODEL_PATH))

person_id = None
boat_id = None

for class_id, class_name in model.names.items():
    name = str(class_name).lower()

    if name == "person":
        person_id = int(class_id)

    elif name == "boat":
        boat_id = int(class_id)


if person_id is None:
    raise RuntimeError(
        "El modelo no contiene la clase 'person'."
    )

if boat_id is None:
    raise RuntimeError(
        "El modelo no contiene la clase 'boat'."
    )


ACTIVE_CLASSES = [
    person_id,
    boat_id,
]


print("=" * 78)
print(
    "VALIDATE BATCH MAIN METRICS + TEGRASTATS + WARM-UP "
    "- VALIDACIÓN FUNCIONAL + RENDIMIENTO + RECURSOS"
)
print("=" * 78)
print(f"Modelo: {MODEL_FILENAME}")
print(
    f"Confidence threshold: "
    f"{CONFIDENCE_THRESHOLD}"
)
print(f"Image size: {IMAGE_SIZE}")
print(f"Clips planificados: {len(plan_rows)}")
print("=" * 78)


# El calentamiento se ejecuta deliberadamente antes de iniciar tegrastats.
# Por tanto, ninguna muestra de recursos ni métrica temporal de la campaña
# incluye el transitorio de inicialización del primer uso del modelo.
warmup_model(
    model=model,
    active_classes=ACTIVE_CLASSES,
    iterations=WARMUP_ITERATIONS,
)


# ============================================================
# INICIAR TEGRASTATS
# ============================================================

resource_monitor = TegraStatsMonitor(
    interval_ms=TEGRASTATS_INTERVAL_MS,
)

resource_monitor.start()


# ============================================================
# VALIDACIÓN POR CLIP
# ============================================================

results_rows = []

try:
    for index, plan in enumerate(
        plan_rows,
        start=1,
    ):
        clip_id = plan["clip_id"].strip()
        area = plan["area"].strip().lower()
        source_video = plan["source_video"].strip()
        scenario = plan["scenario"].strip()

        clip_path = (
            CLIPS_DIR
            / area
            / f"{clip_id}.mp4"
        )

        print(
            f"[{index:02d}/{len(plan_rows):02d}] "
            f"{clip_id} ... ",
            end="",
            flush=True,
        )

        if not clip_path.exists():
            print("ERROR FILE_NOT_FOUND")
            continue

        scene = get_scene_config(clip_path)

        road_roi = scene.get("road_roi")
        road_line = scene.get("road_line")
        water_roi = scene.get("water_roi")
        restricted_side = scene.get("restricted_side")
        authorized_time = scene.get(
            "authorized_time",
            True,
        )

        try:
            reader = LatestFrameReader(clip_path)
        except RuntimeError as exc:
            print(f"ERROR {exc}")
            continue

        video_fps = reader.fps
        source_frame_count = reader.source_frame_count
        source_duration = reader.source_duration_s

        threat_logic = ThreatLogic(
            group_threshold=GROUP_THRESHOLD,
            group_persistence=GROUP_PERSISTENCE,
            boat_persistence=BOAT_PERSISTENCE,
            out_of_hours_persistence=OUT_OF_HOURS_PERSISTENCE,
            restricted_zone_persistence=RESTRICTED_ZONE_PERSISTENCE,
            rearm_persistence=REARM_PERSISTENCE,
        )

        detected = {
            "group": False,
            "restricted": False,
            "out_of_hours": False,
            "boat": False,
        }

        condition_start = {
            "group": None,
            "restricted": None,
            "out_of_hours": None,
            "boat": None,
        }

        first_event_runtime = {
            "group": None,
            "restricted": None,
            "out_of_hours": None,
            "boat": None,
        }

        first_event_latency = {
            "group": None,
            "restricted": None,
            "out_of_hours": None,
            "boat": None,
        }

        max_persons = 0
        max_restricted = 0
        max_boats = 0

        processed_frames = 0
        dropped_frames_by_id = 0
        last_processed_id = None

        inference_times_ms = []

        # Reinicia predictor entre clips para reducir arrastre de estado.
        model.predictor = None

        # Ventana temporal que luego se usa para extraer las muestras de
        # tegrastats pertenecientes exclusivamente a este clip.
        resource_clip_start = time.perf_counter()
        runtime_start = time.perf_counter()
        reader.start()

        try:
            while True:
                (
                    frame_id,
                    frame,
                    capture_ended,
                    frames_captured_now,
                ) = reader.get_latest()

                if frame is None:
                    if capture_ended:
                        break

                    time.sleep(0.001)
                    continue

                if (
                    last_processed_id is not None
                    and frame_id == last_processed_id
                ):
                    if capture_ended:
                        break

                    time.sleep(0.001)
                    continue

                if last_processed_id is None:
                    dropped_frames_by_id += max(0, frame_id)
                else:
                    dropped_frames_by_id += max(
                        0,
                        frame_id - last_processed_id - 1,
                    )

                last_processed_id = frame_id
                processed_frames += 1

                inference_start = time.perf_counter()

                result = model.predict(
                    source=frame,
                    conf=CONFIDENCE_THRESHOLD,
                    imgsz=IMAGE_SIZE,
                    classes=ACTIVE_CLASSES,
                    verbose=False,
                )[0]

                inference_ms = (
                    time.perf_counter()
                    - inference_start
                ) * 1000.0

                inference_times_ms.append(inference_ms)

                runtime_time = (
                    time.perf_counter()
                    - runtime_start
                )

                persons_in_road = 0
                persons_in_restricted = 0
                boats_in_water = 0

                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])

                        class_name = str(
                            model.names[class_id]
                        ).lower()

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0].tolist(),
                        )

                        center = (
                            int((x1 + x2) / 2),
                            int((y1 + y2) / 2),
                        )

                        bottom_center = (
                            int((x1 + x2) / 2),
                            y2,
                        )

                        if class_name == "person":
                            if point_inside_roi(
                                bottom_center,
                                road_roi,
                            ):
                                persons_in_road += 1

                                if (
                                    restricted_side
                                    is not None
                                    and get_line_side(
                                        bottom_center,
                                        road_line,
                                    )
                                    == restricted_side
                                ):
                                    persons_in_restricted += 1

                        elif class_name == "boat":
                            if point_inside_roi(
                                center,
                                water_roi,
                            ):
                                boats_in_water += 1

                max_persons = max(
                    max_persons,
                    persons_in_road,
                )

                max_restricted = max(
                    max_restricted,
                    persons_in_restricted,
                )

                max_boats = max(
                    max_boats,
                    boats_in_water,
                )

                raw_condition = {
                    "group": (
                        persons_in_road
                        > GROUP_THRESHOLD
                    ),
                    "restricted": (
                        persons_in_restricted > 0
                    ),
                    "out_of_hours": (
                        persons_in_road > 0
                        and not authorized_time
                    ),
                    "boat": (
                        boats_in_water > 0
                    ),
                }

                for rule_name, active in raw_condition.items():
                    if active:
                        if condition_start[rule_name] is None:
                            condition_start[rule_name] = runtime_time
                    else:
                        condition_start[rule_name] = None

                group_result = (
                    threat_logic.evaluate_group(
                        persons_in_road,
                        runtime_time,
                    )
                )

                boat_result = (
                    threat_logic.evaluate_boat(
                        boats_in_water,
                        runtime_time,
                    )
                )

                out_result = (
                    threat_logic.evaluate_out_of_hours(
                        persons_in_road,
                        authorized_time,
                        runtime_time,
                    )
                )

                restricted_result = (
                    threat_logic.evaluate_restricted_zone(
                        persons_in_restricted,
                        runtime_time,
                    )
                )

                rule_results = {
                    "group": group_result,
                    "restricted": restricted_result,
                    "out_of_hours": out_result,
                    "boat": boat_result,
                }

                for rule_name, rule_result in rule_results.items():
                    if rule_result["event"]:
                        detected[rule_name] = True

                        if first_event_runtime[rule_name] is None:
                            first_event_runtime[rule_name] = runtime_time

                            if (
                                condition_start[rule_name]
                                is not None
                            ):
                                first_event_latency[rule_name] = (
                                    runtime_time
                                    - condition_start[rule_name]
                                )

        finally:
            reader.stop()

        wall_elapsed = (
            time.perf_counter()
            - runtime_start
        )

        resource_clip_end = time.perf_counter()

        resource_summary = resource_monitor.summarize_window(
            resource_clip_start,
            resource_clip_end,
        )

        frames_captured = reader.frames_captured

        dropped_frames_from_totals = max(
            0,
            frames_captured - processed_frames,
        )

        dropped_frames = dropped_frames_from_totals

        drop_pct = (
            100.0
            * dropped_frames
            / frames_captured
            if frames_captured > 0
            else 0.0
        )

        effective_fps = (
            processed_frames
            / wall_elapsed
            if wall_elapsed > 0
            else math.nan
        )

        realtime_ratio = (
            wall_elapsed
            / source_duration
            if (
                source_duration > 0
                and not math.isnan(source_duration)
            )
            else math.nan
        )

        mean_inference_ms = mean(
            inference_times_ms
        )

        p95_inference_ms = (
            float(
                np.percentile(
                    inference_times_ms,
                    95,
                )
            )
            if inference_times_ms
            else math.nan
        )

        expected = {
            "group": parse_expected(
                plan.get("expected_group")
            ),
            "restricted": parse_expected(
                plan.get("expected_restricted")
            ),
            "out_of_hours": parse_expected(
                plan.get("expected_out_of_hours")
            ),
            "boat": parse_expected(
                plan.get("expected_boat")
            ),
        }

        passes = {
            rule: evaluate_pass(
                expected[rule],
                detected[rule],
            )
            for rule in expected
        }

        applicable = [
            value
            for value in passes.values()
            if value is not None
        ]

        clip_pass = (
            all(applicable)
            if applicable
            else True
        )

        row = {
            "clip_id": clip_id,
            "area": area,
            "source_video": source_video,
            "scenario": scenario,

            "fps_source": f"{video_fps:.3f}",
            "source_frames_total": source_frame_count,
            "source_duration_s": f"{source_duration:.3f}",

            "wall_time_s": f"{wall_elapsed:.3f}",
            "realtime_ratio_wall_over_source": (
                f"{realtime_ratio:.3f}"
                if not math.isnan(realtime_ratio)
                else ""
            ),

            "frames_captured": frames_captured,
            "frames_processed": processed_frames,
            "frames_dropped": dropped_frames,
            "frames_dropped_by_id": dropped_frames_by_id,
            "drop_percentage": f"{drop_pct:.3f}",
            "effective_analysis_fps": f"{effective_fps:.3f}",

            "mean_inference_ms": (
                ""
                if math.isnan(mean_inference_ms)
                else f"{mean_inference_ms:.3f}"
            ),
            "p95_inference_ms": (
                ""
                if math.isnan(p95_inference_ms)
                else f"{p95_inference_ms:.3f}"
            ),

            # Recursos de plataforma obtenidos por tegrastats.
            "resource_samples": resource_summary[
                "resource_samples"
            ],
            "cpu_util_pct_mean": csv_float(
                resource_summary["cpu_util_pct_mean"]
            ),
            "cpu_util_pct_std": csv_float(
                resource_summary["cpu_util_pct_std"]
            ),
            "cpu_util_pct_min": csv_float(
                resource_summary["cpu_util_pct_min"]
            ),
            "cpu_util_pct_max": csv_float(
                resource_summary["cpu_util_pct_max"]
            ),
            "gpu_util_pct_mean": csv_float(
                resource_summary["gpu_util_pct_mean"]
            ),
            "gpu_util_pct_std": csv_float(
                resource_summary["gpu_util_pct_std"]
            ),
            "gpu_util_pct_min": csv_float(
                resource_summary["gpu_util_pct_min"]
            ),
            "gpu_util_pct_max": csv_float(
                resource_summary["gpu_util_pct_max"]
            ),
            "ram_used_mb_mean": csv_float(
                resource_summary["ram_used_mb_mean"]
            ),
            "ram_used_mb_std": csv_float(
                resource_summary["ram_used_mb_std"]
            ),
            "ram_used_mb_min": csv_float(
                resource_summary["ram_used_mb_min"]
            ),
            "ram_used_mb_max": csv_float(
                resource_summary["ram_used_mb_max"]
            ),
            "temperature_c_mean": csv_float(
                resource_summary["temperature_c_mean"]
            ),
            "temperature_c_std": csv_float(
                resource_summary["temperature_c_std"]
            ),
            "temperature_c_min": csv_float(
                resource_summary["temperature_c_min"]
            ),
            "temperature_c_max": csv_float(
                resource_summary["temperature_c_max"]
            ),

            "max_persons_in_road": max_persons,
            "max_persons_in_restricted": max_restricted,
            "max_boats_in_water": max_boats,

            "expected_group": bool_to_csv(
                expected["group"]
            ),
            "detected_group": bool_to_csv(
                detected["group"]
            ),
            "group_latency_s": csv_float(
                first_event_latency["group"]
            ),
            "pass_group": bool_to_csv(
                passes["group"]
            ),

            "expected_restricted": bool_to_csv(
                expected["restricted"]
            ),
            "detected_restricted": bool_to_csv(
                detected["restricted"]
            ),
            "restricted_latency_s": csv_float(
                first_event_latency["restricted"]
            ),
            "pass_restricted": bool_to_csv(
                passes["restricted"]
            ),

            "expected_out_of_hours": bool_to_csv(
                expected["out_of_hours"]
            ),
            "detected_out_of_hours": bool_to_csv(
                detected["out_of_hours"]
            ),
            "out_of_hours_latency_s": csv_float(
                first_event_latency["out_of_hours"]
            ),
            "pass_out_of_hours": bool_to_csv(
                passes["out_of_hours"]
            ),

            "expected_boat": bool_to_csv(
                expected["boat"]
            ),
            "detected_boat": bool_to_csv(
                detected["boat"]
            ),
            "boat_latency_s": csv_float(
                first_event_latency["boat"]
            ),
            "pass_boat": bool_to_csv(
                passes["boat"]
            ),

            "clip_pass": int(clip_pass),
        }

        results_rows.append(row)

        cpu_text = (
            f"{row['cpu_util_pct_mean']}%"
            if row["cpu_util_pct_mean"] != ""
            else "N/A"
        )
        gpu_text = (
            f"{row['gpu_util_pct_mean']}%"
            if row["gpu_util_pct_mean"] != ""
            else "N/A"
        )
        temp_text = (
            f"{row['temperature_c_max']}C"
            if row["temperature_c_max"] != ""
            else "N/A"
        )

        print(
            "PASS" if clip_pass else "FAIL",
            f"| cap={frames_captured}",
            f"| proc={processed_frames}",
            f"| drop={dropped_frames}",
            f"| {drop_pct:.1f}%",
            f"| fps={effective_fps:.2f}",
            f"| CPU={cpu_text}",
            f"| GPU={gpu_text}",
            f"| Tmax={temp_text}",
        )

finally:
    resource_monitor.stop()


# ============================================================
# GUARDAR RESULTADOS DETALLADOS
# ============================================================

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fieldnames = [
    "clip_id",
    "area",
    "source_video",
    "scenario",

    "fps_source",
    "source_frames_total",
    "source_duration_s",

    "wall_time_s",
    "realtime_ratio_wall_over_source",

    "frames_captured",
    "frames_processed",
    "frames_dropped",
    "frames_dropped_by_id",
    "drop_percentage",
    "effective_analysis_fps",

    "mean_inference_ms",
    "p95_inference_ms",

    "resource_samples",
    "cpu_util_pct_mean",
    "cpu_util_pct_std",
    "cpu_util_pct_min",
    "cpu_util_pct_max",
    "gpu_util_pct_mean",
    "gpu_util_pct_std",
    "gpu_util_pct_min",
    "gpu_util_pct_max",
    "ram_used_mb_mean",
    "ram_used_mb_std",
    "ram_used_mb_min",
    "ram_used_mb_max",
    "temperature_c_mean",
    "temperature_c_std",
    "temperature_c_min",
    "temperature_c_max",

    "max_persons_in_road",
    "max_persons_in_restricted",
    "max_boats_in_water",

    "expected_group",
    "detected_group",
    "group_latency_s",
    "pass_group",

    "expected_restricted",
    "detected_restricted",
    "restricted_latency_s",
    "pass_restricted",

    "expected_out_of_hours",
    "detected_out_of_hours",
    "out_of_hours_latency_s",
    "pass_out_of_hours",

    "expected_boat",
    "detected_boat",
    "boat_latency_s",
    "pass_boat",

    "clip_pass",
]

with RESULTS_PATH.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(results_rows)


# ============================================================
# GUARDAR MUESTRAS CRUDAS DE TEGRASTATS
# ============================================================

resource_samples = resource_monitor.get_samples()

raw_resource_fields = [
    "timestamp_s",
    "cpu_util_pct",
    "gpu_util_pct",
    "ram_used_mb",
    "ram_total_mb",
    "temperature_max_c",
    "cpu_temp_c",
    "gpu_temp_c",
    "raw_line",
]

with RESOURCES_RAW_PATH.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=raw_resource_fields,
    )

    writer.writeheader()

    for sample in resource_samples:
        writer.writerow(
            {
                "timestamp_s": csv_float(
                    sample.get("timestamp_s")
                ),
                "cpu_util_pct": csv_float(
                    sample.get("cpu_util_pct")
                ),
                "gpu_util_pct": csv_float(
                    sample.get("gpu_util_pct")
                ),
                "ram_used_mb": csv_float(
                    sample.get("ram_used_mb")
                ),
                "ram_total_mb": csv_float(
                    sample.get("ram_total_mb")
                ),
                "temperature_max_c": csv_float(
                    sample.get("temperature_max_c")
                ),
                "cpu_temp_c": csv_float(
                    sample.get("cpu_temp_c")
                ),
                "gpu_temp_c": csv_float(
                    sample.get("gpu_temp_c")
                ),
                "raw_line": sample.get(
                    "raw_line",
                    "",
                ),
            }
        )


# ============================================================
# RESUMEN GLOBAL DE RECURSOS
# ============================================================

global_resource_metrics = {
    "CPU utilization (%)": [
        sample.get("cpu_util_pct")
        for sample in resource_samples
    ],
    "GPU utilization (%)": [
        sample.get("gpu_util_pct")
        for sample in resource_samples
    ],
    "RAM used (MB)": [
        sample.get("ram_used_mb")
        for sample in resource_samples
    ],
    "Maximum device temperature (C)": [
        sample.get("temperature_max_c")
        for sample in resource_samples
    ],
}

resource_summary_rows = []

for variable_name, values in global_resource_metrics.items():
    variable_stats = stats(values)

    resource_summary_rows.append(
        {
            "variable": variable_name,
            "n_samples": variable_stats["n"],
            "mean": csv_float(variable_stats["mean"]),
            "std": csv_float(variable_stats["std"]),
            "min": csv_float(variable_stats["min"]),
            "max": csv_float(variable_stats["max"]),
        }
    )

with RESOURCES_SUMMARY_PATH.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "variable",
            "n_samples",
            "mean",
            "std",
            "min",
            "max",
        ],
    )

    writer.writeheader()
    writer.writerows(resource_summary_rows)


# ============================================================
# RESUMEN DE VALIDACIÓN FUNCIONAL
# ============================================================

summary_rows = []

rule_columns = {
    "GROUP_DETECTED": (
        "expected_group",
        "detected_group",
    ),
    "RESTRICTED_ZONE_INTRUSION": (
        "expected_restricted",
        "detected_restricted",
    ),
    "OUT_OF_HOURS": (
        "expected_out_of_hours",
        "detected_out_of_hours",
    ),
    "BOAT_INTRUSION": (
        "expected_boat",
        "detected_boat",
    ),
}

for rule_name, (
    expected_col,
    detected_col,
) in rule_columns.items():
    tp = tn = fp = fn = 0

    for row in results_rows:
        expected_value = row[expected_col]
        detected_value = row[detected_col]

        if expected_value == "N/A":
            continue

        expected_bool = str(expected_value) == "1"
        detected_bool = str(detected_value) == "1"

        if expected_bool and detected_bool:
            tp += 1

        elif (
            not expected_bool
            and not detected_bool
        ):
            tn += 1

        elif (
            not expected_bool
            and detected_bool
        ):
            fp += 1

        elif (
            expected_bool
            and not detected_bool
        ):
            fn += 1

    evaluable = tp + tn + fp + fn

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else math.nan
    )

    far = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else math.nan
    )

    summary_rows.append(
        {
            "rule": rule_name,
            "evaluable_clips": evaluable,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "recall": (
                ""
                if math.isnan(recall)
                else f"{recall:.4f}"
            ),
            "false_activation_rate": (
                ""
                if math.isnan(far)
                else f"{far:.4f}"
            ),
        }
    )

summary_fields = [
    "rule",
    "evaluable_clips",
    "TP",
    "TN",
    "FP",
    "FN",
    "recall",
    "false_activation_rate",
]

with SUMMARY_PATH.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=summary_fields,
    )

    writer.writeheader()
    writer.writerows(summary_rows)


# ============================================================
# RESUMEN DE RENDIMIENTO POR ESCENARIO
# ============================================================

performance_rows = []

scenario_names = sorted(
    {
        row["scenario"]
        for row in results_rows
    }
)

metric_columns = {
    "fps_source": "fps_source",
    "effective_analysis_fps": "effective_analysis_fps",
    "frames_captured": "frames_captured",
    "frames_processed": "frames_processed",
    "frames_dropped": "frames_dropped",
    "drop_percentage": "drop_percentage",
    "wall_time_s": "wall_time_s",
    "mean_inference_ms": "mean_inference_ms",
    "cpu_util_pct": "cpu_util_pct_mean",
    "gpu_util_pct": "gpu_util_pct_mean",
    "ram_used_mb": "ram_used_mb_mean",
    "temperature_c": "temperature_c_mean",
}

for scenario_name in scenario_names:
    rows = [
        row
        for row in results_rows
        if row["scenario"] == scenario_name
    ]

    summary = {
        "scenario": scenario_name,
        "n_clips": len(rows),
    }

    for output_name, column_name in metric_columns.items():
        values = []

        for row in rows:
            value = row[column_name]

            if value in ("", None):
                continue

            values.append(float(value))

        avg = mean(values)
        std = sample_std(values)

        summary[f"{output_name}_mean"] = (
            ""
            if math.isnan(avg)
            else f"{avg:.3f}"
        )

        summary[f"{output_name}_std"] = (
            ""
            if math.isnan(std)
            else f"{std:.3f}"
        )

    performance_rows.append(summary)


performance_fields = [
    "scenario",
    "n_clips",
]

for output_name in metric_columns:
    performance_fields.extend(
        [
            f"{output_name}_mean",
            f"{output_name}_std",
        ]
    )

with PERFORMANCE_SUMMARY_PATH.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=performance_fields,
    )

    writer.writeheader()
    writer.writerows(performance_rows)


# ============================================================
# CONSOLA
# ============================================================

pass_count = sum(
    row["clip_pass"]
    for row in results_rows
)

print()
print("=" * 78)
print(
    "VALIDATE BATCH MAIN METRICS + TEGRASTATS + WARM-UP FINALIZADO"
)
print("=" * 78)
print(
    f"Clips procesados: "
    f"{len(results_rows)}"
)
print(
    f"Clips PASS: "
    f"{pass_count}"
)
print(
    f"Clips FAIL: "
    f"{len(results_rows) - pass_count}"
)
print(
    f"Muestras tegrastats: "
    f"{len(resource_samples)}"
)
print(
    f"Resultados detallados: "
    f"{RESULTS_PATH}"
)
print(
    f"Resumen funcional: "
    f"{SUMMARY_PATH}"
)
print(
    f"Resumen de rendimiento: "
    f"{PERFORMANCE_SUMMARY_PATH}"
)
print(
    f"Recursos crudos: "
    f"{RESOURCES_RAW_PATH}"
)
print(
    f"Resumen global de recursos: "
    f"{RESOURCES_SUMMARY_PATH}"
)
print("=" * 78)
