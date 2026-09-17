from pathlib import Path
import csv
import re
import shutil
import statistics
import subprocess
import threading
import time


class JetsonMetrics:
    """
    Monitor de recursos para NVIDIA Jetson basado en tegrastats.

    Registra:
        - CPU global normalizada (%)
        - GPU (%)
        - RAM utilizada (MB)
        - RAM total (MB)
        - Temperatura máxima entre las zonas térmicas (°C)
        - Zona térmica correspondiente

    La CPU global se calcula promediando la utilización de todos
    los núcleos reportados por tegrastats. Los núcleos en estado
    'off' se consideran con utilización 0 %.
    """

    def __init__(self, interval_ms=500):
        self.interval_ms = int(interval_ms)

        self.samples = []

        self.process = None
        self.thread = None

        self.start_time = None


    # =========================================================
    # LOCALIZAR TEGRSTATS
    # =========================================================

    @staticmethod
    def _find_tegrastats():

        candidates = [
            "/usr/bin/tegrastats",
            "/home/nvidia/tegrastats"
        ]

        for candidate in candidates:

            path = Path(candidate)

            if path.exists():
                return str(path)

        executable = shutil.which(
            "tegrastats"
        )

        if executable is not None:
            return executable

        raise FileNotFoundError(
            "No se encontró la utilidad tegrastats."
        )


    # =========================================================
    # INICIAR MONITOR
    # =========================================================

    def start(self):

        if (
            self.process is not None
            and self.process.poll() is None
        ):
            raise RuntimeError(
                "JetsonMetrics ya se encuentra ejecutándose."
            )

        tegrastats_path = (
            self._find_tegrastats()
        )

        self.samples = []

        self.start_time = (
            time.perf_counter()
        )

        command = [
            tegrastats_path,
            "--interval",
            str(self.interval_ms)
        ]

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        self.thread = threading.Thread(
            target=self._reader,
            daemon=True
        )

        self.thread.start()


    # =========================================================
    # LECTOR
    # =========================================================

    def _reader(self):

        if self.process is None:
            return

        if self.process.stdout is None:
            return

        for line in self.process.stdout:

            timestamp = (
                time.perf_counter()
                - self.start_time
            )

            sample = self._parse_line(
                line=line,
                timestamp=timestamp
            )

            if sample is not None:

                self.samples.append(
                    sample
                )


    # =========================================================
    # PARSEO
    # =========================================================

    @staticmethod
    def _parse_line(
        line,
        timestamp
    ):

        # -----------------------------------------------------
        # RAM
        # Ejemplo:
        # RAM 1850/7620MB
        # -----------------------------------------------------

        ram_used_mb = None
        ram_total_mb = None

        ram_match = re.search(
            r"RAM\s+(\d+)/(\d+)MB",
            line
        )

        if ram_match is not None:

            ram_used_mb = float(
                ram_match.group(1)
            )

            ram_total_mb = float(
                ram_match.group(2)
            )


        # -----------------------------------------------------
        # CPU
        #
        # Ejemplo:
        # CPU [10%@729,20%@729,off,5%@729,...]
        #
        # Se promedia sobre TODOS los núcleos.
        # off = 0 %
        # -----------------------------------------------------

        cpu_pct = None

        cpu_match = re.search(
            r"CPU\s+\[([^\]]+)\]",
            line
        )

        if cpu_match is not None:

            cpu_values = []

            cpu_parts = (
                cpu_match
                .group(1)
                .split(",")
            )

            for part in cpu_parts:

                token = (
                    part.strip()
                )

                if token.lower() == "off":

                    cpu_values.append(
                        0.0
                    )

                    continue

                value_match = re.search(
                    r"(\d+(?:\.\d+)?)%",
                    token
                )

                if value_match is not None:

                    cpu_values.append(
                        float(
                            value_match.group(1)
                        )
                    )

            if cpu_values:

                cpu_pct = (
                    sum(cpu_values)
                    / len(cpu_values)
                )


        # -----------------------------------------------------
        # GPU
        #
        # Ejemplo:
        # GR3D_FREQ 95%
        # o
        # GR3D_FREQ 95%@[...]
        # -----------------------------------------------------

        gpu_pct = None

        gpu_match = re.search(
            r"GR3D_FREQ\s+(\d+(?:\.\d+)?)%",
            line
        )

        if gpu_match is not None:

            gpu_pct = float(
                gpu_match.group(1)
            )


        # -----------------------------------------------------
        # TEMPERATURAS
        #
        # Ejemplo:
        # cpu@45.0C soc0@43.5C gpu@44.0C
        #
        # Se conserva la temperatura MÁS ALTA de cada muestra.
        # -----------------------------------------------------

        temp_matches = re.findall(
            r"([A-Za-z0-9_-]+)"
            r"@(-?\d+(?:\.\d+)?)C",
            line
        )

        temp_max_c = None
        temp_zone = None

        if temp_matches:

            temperatures = [
                (
                    zone,
                    float(value)
                )
                for zone, value
                in temp_matches
            ]

            temp_zone, temp_max_c = max(
                temperatures,
                key=lambda item: item[1]
            )


        # -----------------------------------------------------
        # DESCARTAR LÍNEAS SIN DATOS
        # -----------------------------------------------------

        if all(
            value is None
            for value in [
                cpu_pct,
                gpu_pct,
                ram_used_mb,
                temp_max_c
            ]
        ):
            return None


        return {
            "time_s": timestamp,
            "cpu_pct": cpu_pct,
            "gpu_pct": gpu_pct,
            "ram_used_mb": ram_used_mb,
            "ram_total_mb": ram_total_mb,
            "temp_max_c": temp_max_c,
            "temp_zone": temp_zone
        }


    # =========================================================
    # DETENER
    # =========================================================

    def stop(self):

        if self.process is None:
            return

        if self.process.poll() is None:

            self.process.terminate()

            try:

                self.process.wait(
                    timeout=2.0
                )

            except subprocess.TimeoutExpired:

                self.process.kill()

                self.process.wait(
                    timeout=2.0
                )

        if (
            self.thread is not None
            and self.thread.is_alive()
        ):

            self.thread.join(
                timeout=2.0
            )


    # =========================================================
    # ESTADÍSTICA AUXILIAR
    # =========================================================

    @staticmethod
    def _stats(values):

        clean_values = [
            float(value)
            for value in values
            if value is not None
        ]

        if not clean_values:

            return {
                "avg": None,
                "std": None,
                "min": None,
                "max": None
            }

        if len(clean_values) > 1:

            std_value = (
                statistics.pstdev(
                    clean_values
                )
            )

        else:

            std_value = 0.0


        return {
            "avg": statistics.mean(
                clean_values
            ),

            "std": std_value,

            "min": min(
                clean_values
            ),

            "max": max(
                clean_values
            )
        }


    # =========================================================
    # RESUMEN
    # =========================================================

    def summary(self):

        cpu_stats = self._stats(
            [
                sample["cpu_pct"]
                for sample in self.samples
            ]
        )

        gpu_stats = self._stats(
            [
                sample["gpu_pct"]
                for sample in self.samples
            ]
        )

        ram_stats = self._stats(
            [
                sample["ram_used_mb"]
                for sample in self.samples
            ]
        )

        temp_stats = self._stats(
            [
                sample["temp_max_c"]
                for sample in self.samples
            ]
        )


        # -----------------------------------------------------
        # RAM TOTAL
        # -----------------------------------------------------

        ram_total_values = [
            sample["ram_total_mb"]
            for sample in self.samples
            if sample["ram_total_mb"]
            is not None
        ]

        ram_total_mb = (
            ram_total_values[-1]
            if ram_total_values
            else None
        )


        # -----------------------------------------------------
        # ZONA DEL PICO TÉRMICO ABSOLUTO
        # -----------------------------------------------------

        temp_samples = [
            sample
            for sample in self.samples
            if sample["temp_max_c"]
            is not None
        ]

        peak_temp_zone = None

        if temp_samples:

            peak_sample = max(
                temp_samples,
                key=lambda sample:
                sample["temp_max_c"]
            )

            peak_temp_zone = (
                peak_sample["temp_zone"]
            )


        return {
            "sample_count": len(
                self.samples
            ),

            "interval_ms": (
                self.interval_ms
            ),

            "cpu": cpu_stats,

            "gpu": gpu_stats,

            "ram": ram_stats,

            "ram_total_mb": ram_total_mb,

            "temp": temp_stats,

            "temp_peak_zone": (
                peak_temp_zone
            )
        }


    # =========================================================
    # GUARDAR MUESTRAS CRUDAS
    # =========================================================

    def save_csv(self, path):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        fieldnames = [
            "time_s",
            "cpu_pct",
            "gpu_pct",
            "ram_used_mb",
            "ram_total_mb",
            "temp_max_c",
            "temp_zone"
        ]

        with path.open(
            "w",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for sample in self.samples:

                writer.writerow(
                    sample
                )