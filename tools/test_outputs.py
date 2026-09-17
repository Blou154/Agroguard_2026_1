import time
import Jetson.GPIO as GPIO


# ============================================================
# PINES FÍSICOS DEL HEADER J12
# ============================================================

LED_PIN = 29
BUZZER_PIN = 31


# ============================================================
# CONFIGURACIÓN GPIO
# ============================================================

GPIO.setmode(GPIO.BOARD)

GPIO.setup(
    LED_PIN,
    GPIO.OUT,
    initial=GPIO.LOW
)

GPIO.setup(
    BUZZER_PIN,
    GPIO.OUT,
    initial=GPIO.LOW
)


try:

    # ========================================================
    # PRUEBA 1 - LED
    # ========================================================

    print("LED ON")

    GPIO.output(
        LED_PIN,
        GPIO.HIGH
    )

    time.sleep(3)


    print("LED OFF")

    GPIO.output(
        LED_PIN,
        GPIO.LOW
    )

    time.sleep(1)


    # ========================================================
    # PRUEBA 2 - BUZZER
    # ========================================================

    print("BUZZER ON")

    GPIO.output(
        BUZZER_PIN,
        GPIO.HIGH
    )

    time.sleep(3)


    print("BUZZER OFF")

    GPIO.output(
        BUZZER_PIN,
        GPIO.LOW
    )

    time.sleep(1)


    # ========================================================
    # PRUEBA 3 - AMBOS
    # ========================================================

    print("LED + BUZZER ON")

    GPIO.output(
        LED_PIN,
        GPIO.HIGH
    )

    GPIO.output(
        BUZZER_PIN,
        GPIO.HIGH
    )

    time.sleep(3)


    print("LED + BUZZER OFF")

    GPIO.output(
        LED_PIN,
        GPIO.LOW
    )

    GPIO.output(
        BUZZER_PIN,
        GPIO.LOW
    )


finally:

    # ========================================================
    # APAGADO SEGURO
    # ========================================================

    GPIO.output(
        LED_PIN,
        GPIO.LOW
    )

    GPIO.output(
        BUZZER_PIN,
        GPIO.LOW
    )

    GPIO.cleanup()


print("Prueba finalizada.")