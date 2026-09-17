import time
import Jetson.GPIO as GPIO


BUTTON_PIN = 33


GPIO.setmode(
    GPIO.BOARD
)

GPIO.setup(
    BUTTON_PIN,
    GPIO.IN
)


print(
    "========================================"
)

print(
    "PRUEBA DEL PULSADOR - AGROGUARD"
)

print(
    "========================================"
)

print(
    "Conexion: PULL-UP externo"
)

print(
    "Esperado:"
)

print(
    "Libre      -> GPIO 33 = 1"
)

print(
    "Presionado -> GPIO 33 = 0"
)

print(
    "Ctrl+C para salir"
)

print(
    "========================================"
)


previous_state = None


try:

    while True:

        state = GPIO.input(
            BUTTON_PIN
        )


        if state != previous_state:

            if state == GPIO.LOW:

                print(
                    "[BUTTON] PRESIONADO | GPIO 33 = 0"
                )

            else:

                print(
                    "[BUTTON] LIBRE | GPIO 33 = 1"
                )


            previous_state = state


        time.sleep(
            0.05
        )


except KeyboardInterrupt:

    print(
        "\nPrueba detenida."
    )


finally:

    GPIO.cleanup()


print(
    "GPIO liberado."
)