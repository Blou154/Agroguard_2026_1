import threading
import time

import Jetson.GPIO as GPIO


class ActuatorController:

    # ========================================================
    # PINES FÍSICOS J12 - GPIO.BOARD
    # ========================================================

    LED_PIN = 29
    BUZZER_PIN = 31
    BUTTON_PIN = 33

    # Pull-up externo:
    # libre = HIGH
    # presionado = LOW

    BUTTON_POLL_SECONDS = 0.02
    BUTTON_DEBOUNCE_SECONDS = 0.08
    REARM_SILENCE_SECONDS = 1.0


    def __init__(self):

        GPIO.setmode(GPIO.BOARD)

        GPIO.setup(
            self.LED_PIN,
            GPIO.OUT,
            initial=GPIO.LOW
        )

        GPIO.setup(
            self.BUZZER_PIN,
            GPIO.OUT,
            initial=GPIO.LOW
        )

        GPIO.setup(
            self.BUTTON_PIN,
            GPIO.IN
        )

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self._current_threat_active = False
        self._button_pressed = False
        self._last_button_event_time = 0.0

        self.silenced = False
        self.no_threat_since = None

        self.state = {
            "reflector": False,
            "siren": False,
            "button_pressed": False,
            "silenced": False
        }

        # El botón se vigila en un hilo separado para que una
        # inferencia lenta de YOLO no haga perder una pulsación.
        self._button_thread = threading.Thread(
            target=self._button_monitor,
            daemon=True
        )

        self._button_thread.start()

        print("[GPIO] ActuatorController inicializado")
        print(f"[GPIO] LED     -> pin {self.LED_PIN}")
        print(f"[GPIO] BUZZER  -> pin {self.BUZZER_PIN}")
        print(f"[GPIO] BUTTON  -> pin {self.BUTTON_PIN} (pull-up)")


    # ========================================================
    # BOTÓN
    # ========================================================

    def is_button_pressed(self):

        return (
            GPIO.input(self.BUTTON_PIN)
            == GPIO.LOW
        )


    def _button_monitor(self):

        previous_pressed = False

        while not self._stop_event.is_set():

            pressed = self.is_button_pressed()
            now = time.monotonic()

            with self._lock:

                self._button_pressed = pressed
                self.state["button_pressed"] = pressed

                # Flanco: libre -> presionado
                if (
                    pressed
                    and not previous_pressed
                    and (
                        now - self._last_button_event_time
                        >= self.BUTTON_DEBOUNCE_SECONDS
                    )
                ):

                    self._last_button_event_time = now

                    print("[GPIO] Pulsador detectado.")

                    # Solo silencia una alarma que ya está activa.
                    if self._current_threat_active:

                        self.silenced = True
                        self.no_threat_since = None

                        # Apagado inmediato, independiente del
                        # siguiente ciclo de inferencia.
                        GPIO.output(
                            self.LED_PIN,
                            GPIO.LOW
                        )

                        GPIO.output(
                            self.BUZZER_PIN,
                            GPIO.LOW
                        )

                        self.state["reflector"] = False
                        self.state["siren"] = False
                        self.state["silenced"] = True

                        print(
                            "[GPIO] Alarma silenciada."
                        )

            previous_pressed = pressed

            time.sleep(
                self.BUTTON_POLL_SECONDS
            )


    # ========================================================
    # CONTROL DE SALIDAS
    # ========================================================

    def update(
        self,
        threat_active
    ):

        threat_active = bool(
            threat_active
        )

        now = time.monotonic()

        with self._lock:

            self._current_threat_active = (
                threat_active
            )

            # =================================================
            # REARME
            #
            # Tras un silenciamiento, la amenaza debe estar
            # ausente 1 s continuo para rearmarse.
            # =================================================

            if self.silenced:

                if threat_active:

                    self.no_threat_since = None

                else:

                    if self.no_threat_since is None:

                        self.no_threat_since = now

                    elif (
                        now - self.no_threat_since
                        >= self.REARM_SILENCE_SECONDS
                    ):

                        self.silenced = False
                        self.no_threat_since = None

                        print(
                            "[GPIO] Sistema rearmado."
                        )

            else:

                self.no_threat_since = None


            output_active = (
                threat_active
                and not self.silenced
            )


            GPIO.output(
                self.LED_PIN,
                GPIO.HIGH
                if output_active
                else GPIO.LOW
            )

            GPIO.output(
                self.BUZZER_PIN,
                GPIO.HIGH
                if output_active
                else GPIO.LOW
            )


            self.state["reflector"] = (
                output_active
            )

            self.state["siren"] = (
                output_active
            )

            self.state["button_pressed"] = (
                self._button_pressed
            )

            self.state["silenced"] = (
                self.silenced
            )


    def get_state(self):

        with self._lock:

            return self.state.copy()


    # ========================================================
    # CIERRE
    # ========================================================

    def cleanup(self):

        self._stop_event.set()

        if self._button_thread.is_alive():

            self._button_thread.join(
                timeout=1.0
            )

        with self._lock:

            GPIO.output(
                self.LED_PIN,
                GPIO.LOW
            )

            GPIO.output(
                self.BUZZER_PIN,
                GPIO.LOW
            )

            self.state["reflector"] = False
            self.state["siren"] = False
            self.state["button_pressed"] = False
            self.state["silenced"] = False

        GPIO.cleanup()

        print(
            "[GPIO] Salidas apagadas "
            "y GPIO liberados."
        )