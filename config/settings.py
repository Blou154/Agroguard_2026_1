# ============================================================
# CONFIGURACIÓN GENERAL DE AGROGUARD
# ============================================================


# ============================================================
# DETECTOR
# ============================================================

MODEL_FILENAME = "agroguard_yolov8n_v1.pt"

CONFIDENCE_THRESHOLD = 0.50

IMAGE_SIZE = 640 # 1280

TRACKER = "botsort.yaml"


# ============================================================
# REGLAS DE AMENAZA
# ============================================================

# Se activa GROUP_DETECTED cuando:
# person_count > GROUP_THRESHOLD
GROUP_THRESHOLD = 3


# Tiempo mínimo que debe permanecer una condición
# antes de confirmarse como evento.
GROUP_PERSISTENCE = 1.0

BOAT_PERSISTENCE = 1.0

OUT_OF_HOURS_PERSISTENCE = 1.0

RESTRICTED_ZONE_PERSISTENCE = 1.0


# ============================================================
# REARME
# ============================================================

# Tiempo que una condición debe permanecer ausente
# antes de considerarse terminada.
REARM_PERSISTENCE = 0.5


# ============================================================
# EVIDENCIA
# ============================================================

# Segundos de video anteriores a la confirmación del evento.
PRE_EVENT_SECONDS = 2.0

# Segundos posteriores a la confirmación del evento.
POST_EVENT_SECONDS = 3.0