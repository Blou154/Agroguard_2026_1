class ThreatLogic:

    def __init__(
        self,
        group_threshold=3,
        group_persistence=1.0,
        boat_persistence=1.0,
        out_of_hours_persistence=1.0,
        restricted_zone_persistence=1.0,
        rearm_persistence=0.5
    ):

        # ====================================================
        # PARÁMETROS
        # ====================================================

        self.group_threshold = group_threshold

        self.group_persistence = group_persistence

        self.boat_persistence = boat_persistence

        self.out_of_hours_persistence = (
            out_of_hours_persistence
        )

        self.restricted_zone_persistence = (
            restricted_zone_persistence
        )

        self.rearm_persistence = rearm_persistence


        # ====================================================
        # ESTADO: GRUPO
        # ====================================================

        self.group_start_time = None
        self.group_end_time = None
        self.group_event_active = False


        # ====================================================
        # ESTADO: BOAT
        # ====================================================

        self.boat_start_time = None
        self.boat_end_time = None
        self.boat_event_active = False


        # ====================================================
        # ESTADO: FUERA DE HORARIO
        # ====================================================

        self.out_of_hours_start_time = None
        self.out_of_hours_end_time = None
        self.out_of_hours_event_active = False


        # ====================================================
        # ESTADO: ZONA RESTRINGIDA
        # ====================================================

        self.restricted_zone_start_time = None
        self.restricted_zone_end_time = None
        self.restricted_zone_event_active = False


    # ========================================================
    # FUNCIÓN INTERNA GENÉRICA
    # ========================================================

    def _evaluate_persistent_condition(
        self,
        condition_active,
        timestamp,
        persistence,
        start_attr,
        end_attr,
        active_attr
    ):

        event_detected = False

        start_time = getattr(
            self,
            start_attr
        )

        end_time = getattr(
            self,
            end_attr
        )

        event_active = getattr(
            self,
            active_attr
        )


        # ----------------------------------------------------
        # CONDICIÓN PRESENTE
        # ----------------------------------------------------

        if condition_active:

            # Cancelar intento de rearme
            end_time = None


            # Iniciar temporizador de activación
            if start_time is None:

                start_time = timestamp


            elapsed = (
                timestamp
                - start_time
            )


            # Confirmar evento
            if (
                elapsed >= persistence
                and not event_active
            ):

                event_detected = True

                event_active = True


        # ----------------------------------------------------
        # CONDICIÓN AUSENTE
        # ----------------------------------------------------

        else:

            # Todavía no había evento confirmado
            if not event_active:

                start_time = None

                end_time = None

                elapsed = 0.0


            # Evento confirmado:
            # empezar temporizador de rearme
            else:

                if end_time is None:

                    end_time = timestamp


                absence_elapsed = (
                    timestamp
                    - end_time
                )


                if (
                    absence_elapsed
                    >= self.rearm_persistence
                ):

                    event_active = False

                    start_time = None

                    end_time = None

                    elapsed = 0.0


                else:

                    elapsed = (
                        timestamp
                        - start_time
                    )


        setattr(
            self,
            start_attr,
            start_time
        )

        setattr(
            self,
            end_attr,
            end_time
        )

        setattr(
            self,
            active_attr,
            event_active
        )


        return {
            "event": event_detected,
            "condition_active": condition_active,
            "event_active": event_active,
            "elapsed": elapsed
        }


    # ========================================================
    # REGLA 1
    # GRUPO > 3 PERSONAS
    # ========================================================

    def evaluate_group(
        self,
        person_count,
        timestamp
    ):

        result = (
            self._evaluate_persistent_condition(
                condition_active=(
                    person_count
                    > self.group_threshold
                ),
                timestamp=timestamp,
                persistence=(
                    self.group_persistence
                ),
                start_attr="group_start_time",
                end_attr="group_end_time",
                active_attr="group_event_active"
            )
        )

        result["person_count"] = (
            person_count
        )

        return result


    # ========================================================
    # REGLA 2
    # EMBARCACIÓN EN WATER ROI
    # ========================================================

    def evaluate_boat(
        self,
        boat_count,
        timestamp
    ):

        result = (
            self._evaluate_persistent_condition(
                condition_active=(
                    boat_count > 0
                ),
                timestamp=timestamp,
                persistence=(
                    self.boat_persistence
                ),
                start_attr="boat_start_time",
                end_attr="boat_end_time",
                active_attr="boat_event_active"
            )
        )

        result["boat_count"] = (
            boat_count
        )

        return result


    # ========================================================
    # REGLA 3
    # PERSONA FUERA DE HORARIO
    # ========================================================

    def evaluate_out_of_hours(
        self,
        person_count,
        authorized_time,
        timestamp
    ):

        condition_active = (
            person_count > 0
            and not authorized_time
        )

        result = (
            self._evaluate_persistent_condition(
                condition_active=condition_active,
                timestamp=timestamp,
                persistence=(
                    self.out_of_hours_persistence
                ),
                start_attr=(
                    "out_of_hours_start_time"
                ),
                end_attr=(
                    "out_of_hours_end_time"
                ),
                active_attr=(
                    "out_of_hours_event_active"
                )
            )
        )

        result["person_count"] = (
            person_count
        )

        result["authorized_time"] = (
            authorized_time
        )

        return result


    # ========================================================
    # REGLA 4
    # PERSONA EN ZONA RESTRINGIDA
    # ========================================================

    def evaluate_restricted_zone(
        self,
        person_count,
        timestamp
    ):

        result = (
            self._evaluate_persistent_condition(
                condition_active=(
                    person_count > 0
                ),
                timestamp=timestamp,
                persistence=(
                    self.restricted_zone_persistence
                ),
                start_attr=(
                    "restricted_zone_start_time"
                ),
                end_attr=(
                    "restricted_zone_end_time"
                ),
                active_attr=(
                    "restricted_zone_event_active"
                )
            )
        )

        result["person_count"] = (
            person_count
        )

        return result