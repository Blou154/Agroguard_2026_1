from pathlib import Path
import argparse
import csv
import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]

VALIDATION_DIR = ROOT_DIR / "data" / "validation"
CLIPS_DIR = VALIDATION_DIR / "clips"
MANIFEST_PATH = VALIDATION_DIR / "validation_plan.csv"

CLIPS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONVERTIR TIEMPO A SEGUNDOS
# ============================================================

def parse_time(value: str) -> float:
    """
    Permite escribir:

    15.5
    01:20
    00:01:20.5
    """

    value = value.strip()

    if ":" not in value:
        return float(value)

    parts = list(
        map(
            float,
            value.split(":")
        )
    )

    if len(parts) == 2:

        minutes, seconds = parts

        return (
            minutes * 60
            + seconds
        )

    if len(parts) == 3:

        hours, minutes, seconds = parts

        return (
            hours * 3600
            + minutes * 60
            + seconds
        )

    raise ValueError(
        f"Formato de tiempo inválido: {value}"
    )


# ============================================================
# RECORTAR VIDEO
# ============================================================

def cut_video(
    source_path,
    output_path,
    start_seconds,
    end_seconds
):

    cap = cv2.VideoCapture(
        str(source_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"No se pudo abrir:\n"
            f"{source_path}"
        )


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    duration = (
        total_frames / fps
    )


    if start_seconds < 0:

        raise ValueError(
            "El tiempo inicial no puede ser negativo."
        )


    if end_seconds <= start_seconds:

        raise ValueError(
            "El tiempo final debe ser mayor "
            "que el tiempo inicial."
        )


    if start_seconds >= duration:

        raise ValueError(
            "El tiempo inicial supera "
            "la duración del video."
        )


    end_seconds = min(
        end_seconds,
        duration
    )


    start_frame = round(
        start_seconds * fps
    )

    end_frame = round(
        end_seconds * fps
    )


    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )


    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )


    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (
            width,
            height
        )
    )


    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            f"No se pudo crear:\n"
            f"{output_path}"
        )


    current_frame = (
        start_frame
    )


    written_frames = 0


    while (
        current_frame
        < end_frame
    ):

        ret, frame = (
            cap.read()
        )


        if not ret:

            break


        writer.write(
            frame
        )


        written_frames += 1

        current_frame += 1


    cap.release()

    writer.release()


    clip_duration = (
        written_frames / fps
    )


    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frames": written_frames,
        "duration": clip_duration
    }


# ============================================================
# REGISTRAR CLIP EN CSV
# ============================================================

def register_clip(
    clip_id,
    source,
    output,
    start_seconds,
    end_seconds,
    scenario,
    expected_event,
    expected_positive,
    notes,
    metadata
):

    file_exists = (
        MANIFEST_PATH.exists()
    )


    fields = [
        "clip_id",
        "source",
        "clip",
        "start_s",
        "end_s",
        "duration_s",
        "scenario",
        "expected_event",
        "expected_positive",
        "fps",
        "width",
        "height",
        "frames",
        "notes"
    ]


    with open(
        MANIFEST_PATH,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )


        if not file_exists:

            writer.writeheader()


        writer.writerow({

            "clip_id":
                clip_id,

            "source":
                str(source),

            "clip":
                str(output),

            "start_s":
                start_seconds,

            "end_s":
                end_seconds,

            "duration_s":
                round(
                    metadata["duration"],
                    3
                ),

            "scenario":
                scenario,

            "expected_event":
                expected_event,

            "expected_positive":
                int(
                    expected_positive
                ),

            "fps":
                metadata["fps"],

            "width":
                metadata["width"],

            "height":
                metadata["height"],

            "frames":
                metadata["frames"],

            "notes":
                notes
        })


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Recorta clips para la validación "
            "experimental de AgroGuard."
        )
    )


    parser.add_argument(
        "--source",
        required=True,
        help="Ruta del video original."
    )


    parser.add_argument(
        "--start",
        required=True,
        help="Tiempo inicial. Ej.: 12.5 o 01:20."
    )


    parser.add_argument(
        "--end",
        required=True,
        help="Tiempo final."
    )


    parser.add_argument(
        "--id",
        required=True,
        help="Identificador único del clip."
    )


    parser.add_argument(
        "--scenario",
        required=True,
        choices=[
            "group",
            "out_of_hours",
            "restricted_zone",
            "boat",
            "negative"
        ]
    )


    parser.add_argument(
        "--expected-event",
        default="NONE",
        choices=[
            "GROUP_DETECTED",
            "OUT_OF_HOURS",
            "RESTRICTED_ZONE_INTRUSION",
            "BOAT_INTRUSION",
            "NONE"
        ]
    )


    parser.add_argument(
        "--positive",
        type=int,
        choices=[
            0,
            1
        ],
        default=1,
        help=(
            "1 si debe producirse el evento, "
            "0 si no debe producirse."
        )
    )


    parser.add_argument(
        "--notes",
        default=""
    )


    args = (
        parser.parse_args()
    )


    source_path = Path(
        args.source
    )


    if not source_path.is_absolute():

        source_path = (
            ROOT_DIR
            / source_path
        )


    if not source_path.exists():

        raise FileNotFoundError(
            source_path
        )


    start_seconds = (
        parse_time(
            args.start
        )
    )


    end_seconds = (
        parse_time(
            args.end
        )
    )


    output_path = (
        CLIPS_DIR
        / f"{args.id}.mp4"
    )


    print(
        "\nRecortando..."
    )


    metadata = cut_video(
        source_path,
        output_path,
        start_seconds,
        end_seconds
    )


    register_clip(
        clip_id=args.id,
        source=source_path,
        output=output_path,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        scenario=args.scenario,
        expected_event=args.expected_event,
        expected_positive=bool(
            args.positive
        ),
        notes=args.notes,
        metadata=metadata
    )


    print(
        "\nCLIP GENERADO"
    )

    print(
        "Archivo:",
        output_path
    )

    print(
        "Duración:",
        round(
            metadata["duration"],
            2
        ),
        "s"
    )

    print(
        "FPS:",
        round(
            metadata["fps"],
            3
        )
    )

    print(
        "Frames:",
        metadata["frames"]
    )

    print(
        "Evento esperado:",
        args.expected_event
    )

    print(
        "\nRegistro actualizado:"
    )

    print(
        MANIFEST_PATH
    )


if __name__ == "__main__":

    main()