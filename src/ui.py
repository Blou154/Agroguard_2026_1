import cv2


# ============================================================
# CONFIGURACIÓN VISUAL
# ============================================================

FONT = cv2.FONT_HERSHEY_SIMPLEX

FONT_SCALE = 0.60
FONT_SCALE_TITLE = 0.72

TEXT_THICKNESS = 1
TITLE_THICKNESS = 2

LINE_HEIGHT = 30

PANEL_X = 15
PANEL_Y = 15
PANEL_WIDTH = 410

PADDING = 15


# ============================================================
# COLORES BGR
# ============================================================

COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 220, 0)
COLOR_RED = (0, 0, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_GRAY = (180, 180, 180)

PANEL_COLOR = (20, 20, 20)


# ============================================================
# RECTÁNGULO SEMITRANSPARENTE
# ============================================================

def draw_transparent_rectangle(
    frame,
    x1,
    y1,
    x2,
    y2,
    color,
    alpha=0.70
):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        color,
        -1
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame
    )


# ============================================================
# FILA DEL PANEL
# ============================================================

def draw_row(
    frame,
    label,
    value,
    y,
    value_color=COLOR_WHITE
):

    label_x = (
        PANEL_X
        + PADDING
    )

    value_right = (
        PANEL_X
        + PANEL_WIDTH
        - PADDING
    )


    # Etiqueta
    cv2.putText(
        frame,
        label,
        (label_x, y),
        FONT,
        FONT_SCALE,
        COLOR_WHITE,
        TEXT_THICKNESS,
        cv2.LINE_AA
    )


    # Calcular ancho del valor para alinearlo a la derecha
    value_size, _ = cv2.getTextSize(
        str(value),
        FONT,
        FONT_SCALE,
        TEXT_THICKNESS
    )

    value_x = (
        value_right
        - value_size[0]
    )


    cv2.putText(
        frame,
        str(value),
        (value_x, y),
        FONT,
        FONT_SCALE,
        value_color,
        TEXT_THICKNESS,
        cv2.LINE_AA
    )


# ============================================================
# PANEL PRINCIPAL
# ============================================================

def draw_dashboard(
    frame,
    persons_in_road,
    persons_in_restricted_zone,
    boats_in_water,
    video_time,
    authorized_time,
    group_active,
    boat_active,
    out_of_hours_active,
    restricted_active,
    reflector_on,
    siren_on
):

    # ========================================================
    # IDENTIFICAR EVENTOS ACTIVOS
    # ========================================================

    active_events = []

    if group_active:
        active_events.append(
            "GROUP_DETECTED"
        )

    if boat_active:
        active_events.append(
            "BOAT_INTRUSION"
        )

    if out_of_hours_active:
        active_events.append(
            "OUT_OF_HOURS"
        )

    if restricted_active:
        active_events.append(
            "RESTRICTED_ZONE_INTRUSION"
        )


    threat_active = (
        len(active_events) > 0
    )


    # ========================================================
    # CALCULAR ALTURA DEL PANEL
    # ========================================================

    base_rows = 9

    extra_rows = len(
        active_events
    )

    panel_height = (
        45
        + (base_rows + extra_rows)
        * LINE_HEIGHT
    )


    # ========================================================
    # FONDO
    # ========================================================

    draw_transparent_rectangle(
        frame,
        PANEL_X,
        PANEL_Y,
        PANEL_X + PANEL_WIDTH,
        PANEL_Y + panel_height,
        PANEL_COLOR,
        alpha=0.72
    )


    # ========================================================
    # TÍTULO
    # ========================================================

    y = (
        PANEL_Y
        + 32
    )

    cv2.putText(
        frame,
        "AGROGUARD AI",
        (
            PANEL_X
            + PADDING,
            y
        ),
        FONT,
        FONT_SCALE_TITLE,
        COLOR_WHITE,
        TITLE_THICKNESS,
        cv2.LINE_AA
    )


    # Línea separadora
    cv2.line(
        frame,
        (
            PANEL_X + PADDING,
            y + 12
        ),
        (
            PANEL_X
            + PANEL_WIDTH
            - PADDING,
            y + 12
        ),
        COLOR_GRAY,
        1
    )


    # ========================================================
    # DATOS
    # ========================================================

    y += 48

    draw_row(
        frame,
        "Personas camino",
        persons_in_road,
        y
    )


    y += LINE_HEIGHT

    draw_row(
        frame,
        "Personas restringidas",
        persons_in_restricted_zone,
        y,
        (
            COLOR_RED
            if persons_in_restricted_zone > 0
            else COLOR_WHITE
        )
    )


    y += LINE_HEIGHT

    draw_row(
        frame,
        "Embarcaciones",
        boats_in_water,
        y,
        (
            COLOR_ORANGE
            if boats_in_water > 0
            else COLOR_WHITE
        )
    )


    y += LINE_HEIGHT

    draw_row(
        frame,
        "Tiempo",
        f"{video_time:.2f} s",
        y
    )


    y += LINE_HEIGHT


    schedule_text = (
        "AUTORIZADO"
        if authorized_time
        else "NO AUTORIZADO"
    )


    draw_row(
        frame,
        "Horario",
        schedule_text,
        y,
        (
            COLOR_GREEN
            if authorized_time
            else COLOR_ORANGE
        )
    )


    # ========================================================
    # SEPARADOR
    # ========================================================

    y += 15

    cv2.line(
        frame,
        (
            PANEL_X + PADDING,
            y
        ),
        (
            PANEL_X
            + PANEL_WIDTH
            - PADDING,
            y
        ),
        COLOR_GRAY,
        1
    )


    # ========================================================
    # ESTADO GENERAL
    # ========================================================

    y += LINE_HEIGHT


    draw_row(
        frame,
        "Estado",
        (
            "ALERTA"
            if threat_active
            else "SEGURO"
        ),
        y,
        (
            COLOR_RED
            if threat_active
            else COLOR_GREEN
        )
    )


    # ========================================================
    # EVENTOS ACTIVOS
    # ========================================================

    for event_name in active_events:

        y += LINE_HEIGHT

        cv2.putText(
            frame,
            event_name,
            (
                PANEL_X
                + PADDING,
                y
            ),
            FONT,
            FONT_SCALE,
            COLOR_RED,
            TEXT_THICKNESS,
            cv2.LINE_AA
        )


    # ========================================================
    # ACTUADORES
    # ========================================================

    y += LINE_HEIGHT

    draw_row(
        frame,
        "Reflector",
        (
            "ON"
            if reflector_on
            else "OFF"
        ),
        y,
        (
            COLOR_RED
            if reflector_on
            else COLOR_GREEN
        )
    )


    y += LINE_HEIGHT

    draw_row(
        frame,
        "Sirena",
        (
            "ON"
            if siren_on
            else "OFF"
        ),
        y,
        (
            COLOR_RED
            if siren_on
            else COLOR_GREEN
        )
    )