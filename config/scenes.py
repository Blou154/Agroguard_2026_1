from copy import deepcopy
from pathlib import Path


# ============================================================
# CONFIGURACIÓN CAMINO
# ============================================================

ROAD_PROFILE = {

    "road_roi": [
        (3, 789),
        (2, 231),
        (461, 224),
        (463, 789)
    ],

    "road_line": [
        (4, 609),
        (263, 317)
    ],

    "restricted_side": -1,

    "water_roi": None,

    "authorized_time": True
}


# ============================================================
# ROI AGUA - VIDEO 3
# ============================================================

WATER_ROI_VIDEO_3 = [
    (1278, 349),
    (1, 342),
    (2, 716),
    (1279, 717)
]


# ============================================================
# ROI AGUA - DEMÁS VIDEOS
# ============================================================

WATER_ROI_OTHER = [
    (829, 456),
    (2, 456),
    (2, 205),
    (829, 205)
]


# ============================================================
# PERFIL AGUA - VIDEO 3
# ============================================================

WATER_PROFILE_VIDEO_3 = {

    "road_roi": None,

    "road_line": None,

    "restricted_side": None,

    "water_roi": WATER_ROI_VIDEO_3,

    "authorized_time": True
}


# ============================================================
# PERFIL AGUA - OTROS VIDEOS
# ============================================================

WATER_PROFILE_OTHER = {

    "road_roi": None,

    "road_line": None,

    "restricted_side": None,

    "water_roi": WATER_ROI_OTHER,

    "authorized_time": True
}


# ============================================================
# OBTENER CONFIGURACIÓN SEGÚN VIDEO
# ============================================================

def get_scene_config(source_path):

    filename = (
        Path(source_path)
        .name
        .lower()
    )


    # ========================================================
    # VIDEO ORIGINAL 003
    # ========================================================

    if filename == "vid_agu_003.mp4":

        return deepcopy(
            WATER_PROFILE_VIDEO_3
        )


    # ========================================================
    # CLIPS PROCEDENTES DEL VIDEO 003
    #
    # WATER_BOAT_004 hasta WATER_BOAT_009
    # ========================================================

    video_3_clips = {
        "water_boat_004.mp4",
        "water_boat_005.mp4",
        "water_boat_006.mp4",
        "water_boat_007.mp4",
        "water_boat_008.mp4",
        "water_boat_009.mp4"
    }


    if filename in video_3_clips:

        return deepcopy(
            WATER_PROFILE_VIDEO_3
        )


    # ========================================================
    # OTROS VIDEOS ORIGINALES DE AGUA
    # ========================================================

    if filename in {
        "vid_agu_001.mp4",
        "vid_agu_004.mp4",
        "lago_01.mp4"
    }:

        return deepcopy(
            WATER_PROFILE_OTHER
        )


    # ========================================================
    # RESTO DE CLIPS DE AGUA
    #
    # Incluye:
    # WATER_NEG_001 ... WATER_NEG_004
    # WATER_BOAT_001 ... WATER_BOAT_003
    # WATER_BOAT_010 ... WATER_BOAT_015
    # ========================================================

    if filename.startswith(
        "water_"
    ):

        return deepcopy(
            WATER_PROFILE_OTHER
        )


    # ========================================================
    # VIDEOS / CLIPS DE CAMINO
    # ========================================================

    if (
        filename.startswith("road_")
        or filename.startswith("vid_cam_")
        or filename == "camino_01.mp4"
    ):

        return deepcopy(
            ROAD_PROFILE
        )


    # ========================================================
    # VIDEO NO RECONOCIDO
    # ========================================================

    raise KeyError(
        f"No existe configuración "
        f"para el video: {filename}"
    )