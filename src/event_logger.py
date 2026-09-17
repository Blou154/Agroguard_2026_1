from pathlib import Path
import sqlite3
from datetime import datetime

import cv2


class EventLogger:

    def __init__(self, root_dir):

        self.root_dir = Path(root_dir)

        # ====================================================
        # CARPETAS
        # ====================================================

        self.outputs_dir = (
            self.root_dir
            / "outputs"
        )

        self.evidence_dir = (
            self.root_dir
            / "runs"
            / "events"
        )

        self.database_path = (
            self.outputs_dir
            / "agroguard_events.db"
        )

        self.outputs_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.evidence_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self._initialize_database()


    # ========================================================
    # INICIALIZAR BASE DE DATOS
    # ========================================================

    def _initialize_database(self):

        with sqlite3.connect(
            self.database_path
        ) as connection:

            cursor = connection.cursor()

            # ------------------------------------------------
            # Crear tabla si todavía no existe
            # ------------------------------------------------

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    scene TEXT NOT NULL,
                    video_time REAL,
                    frame_number INTEGER,
                    event_type TEXT NOT NULL,
                    person_count INTEGER DEFAULT 0,
                    boat_count INTEGER DEFAULT 0,
                    evidence_path TEXT,
                    video_path TEXT
                )
                """
            )

            # ------------------------------------------------
            # Compatibilidad con base de datos anterior
            # ------------------------------------------------

            cursor.execute(
                "PRAGMA table_info(events)"
            )

            columns = [
                row[1]
                for row in cursor.fetchall()
            ]

            if "video_path" not in columns:

                cursor.execute(
                    """
                    ALTER TABLE events
                    ADD COLUMN video_path TEXT
                    """
                )

            connection.commit()


    # ========================================================
    # REGISTRAR EVENTO + IMAGEN
    # ========================================================

    def log_event(
        self,
        event_type,
        frame,
        scene,
        video_time=None,
        frame_number=None,
        person_count=0,
        boat_count=0
    ):

        created_at = datetime.now()

        timestamp_text = (
            created_at.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        file_timestamp = (
            created_at.strftime(
                "%Y%m%d_%H%M%S_%f"
            )
        )


        # ====================================================
        # NOMBRE BASE DE LAS EVIDENCIAS
        # ====================================================

        base_filename = (
            f"{file_timestamp}_"
            f"{event_type}"
        )


        # ====================================================
        # GUARDAR IMAGEN
        # ====================================================

        image_path = (
            self.evidence_dir
            / f"{base_filename}.jpg"
        )

        image_saved = cv2.imwrite(
            str(image_path),
            frame
        )

        if not image_saved:

            raise RuntimeError(
                "No se pudo guardar "
                "la imagen del evento."
            )


        # ====================================================
        # RUTA FUTURA DEL VIDEO
        # ====================================================

        video_path = (
            self.evidence_dir
            / f"{base_filename}.mp4"
        )


        # ====================================================
        # GUARDAR EVENTO EN SQLITE
        # ====================================================

        with sqlite3.connect(
            self.database_path
        ) as connection:

            cursor = connection.cursor()

            cursor.execute(
                """
                INSERT INTO events (
                    created_at,
                    scene,
                    video_time,
                    frame_number,
                    event_type,
                    person_count,
                    boat_count,
                    evidence_path,
                    video_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp_text,
                    scene,
                    video_time,
                    frame_number,
                    event_type,
                    person_count,
                    boat_count,
                    str(image_path),
                    None
                )
            )

            event_id = cursor.lastrowid

            connection.commit()


        return {
            "event_id": event_id,
            "created_at": timestamp_text,
            "event_type": event_type,
            "image_path": str(image_path),
            "video_path": str(video_path)
        }


    # ========================================================
    # GUARDAR CLIP DE VIDEO
    # ========================================================

    def save_event_video(
        self,
        event_id,
        video_path,
        frames,
        fps
    ):

        if not frames:

            print(
                f"[ADVERTENCIA] "
                f"No existen frames para "
                f"el video del evento {event_id}."
            )

            return None


        first_frame = frames[0]

        height, width = (
            first_frame.shape[:2]
        )


        # ====================================================
        # CODEC MP4
        # ====================================================

        fourcc = (
            cv2.VideoWriter_fourcc(
                *"mp4v"
            )
        )


        writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            fps,
            (
                width,
                height
            )
        )


        if not writer.isOpened():

            raise RuntimeError(
                f"No se pudo crear "
                f"el video:\n{video_path}"
            )


        # ====================================================
        # ESCRIBIR FRAMES
        # ====================================================

        for frame in frames:

            writer.write(
                frame
            )


        writer.release()


        # ====================================================
        # ACTUALIZAR SQLITE
        # ====================================================

        with sqlite3.connect(
            self.database_path
        ) as connection:

            cursor = connection.cursor()

            cursor.execute(
                """
                UPDATE events
                SET video_path = ?
                WHERE id = ?
                """,
                (
                    str(video_path),
                    event_id
                )
            )

            connection.commit()


        print(
            f"[EVIDENCIA] "
            f"Video del evento {event_id} guardado: "
            f"{video_path}"
        )


        return str(video_path)