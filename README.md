# Agroguard_2026_1

Sistema experimental de vigilancia inteligente basado en visión artificial y procesamiento en el borde para detectar personas, agrupaciones, accesos a zonas restringidas y embarcaciones.

Este repositorio contiene el código, la configuración, el modelo y los resultados experimentales desarrollados como parte de una tesis de Ingeniería Mecatrónica.

## Descripción

AgroGuard utiliza un modelo YOLOv8n ajustado para detectar las clases `person` y `boat`. La inferencia se ejecuta localmente en una NVIDIA Jetson Orin Nano y sus resultados se evalúan mediante reglas espaciales y temporales.

Cuando se confirma una condición de amenaza, el sistema puede:

- activar salidas físicas mediante GPIO;
- encender un indicador LED;
- activar un buzzer;
- registrar el evento en una base de datos SQLite;
- almacenar una imagen de evidencia;
- generar un clip con fotogramas anteriores y posteriores al evento.

## Alcance

AgroGuard fue desarrollado y evaluado como una prueba de concepto.

La validación comprende:

- inferencia de YOLOv8n en una NVIDIA Jetson Orin Nano;
- procesamiento acelerado mediante CUDA;
- detección de personas y embarcaciones;
- evaluación de regiones de interés;
- confirmación temporal de amenazas;
- almacenamiento local de eventos;
- generación de evidencias;
- actuación experimental mediante LED y buzzer;
- medición de FPS y latencia;
- monitorización de CPU, GPU, RAM y temperatura.

La implementación no constituye todavía un sistema industrial certificado. La actuación mediante módulos ADAM, relés, reflector y sirena industrial pertenece al diseño de campo y no a la plataforma utilizada durante la validación experimental.

## Reglas de amenaza

### `GROUP_DETECTED`

Se confirma cuando se detectan más de tres personas dentro de la región supervisada durante el intervalo de persistencia configurado.

### `RESTRICTED_ZONE_INTRUSION`

Se confirma cuando al menos una persona permanece en el lado restringido de la escena durante el intervalo configurado.

### `BOAT_INTRUSION`

Se confirma cuando al menos una embarcación permanece dentro de la región acuática supervisada.

### `OUT_OF_HOURS`

Está destinada a detectar personas fuera del horario autorizado.

## Estructura del repositorio

```text
Agroguard/
├── main.py
├── main2.py
├── main_metrics.py
│
├── config/
│   ├── scenes.py
│   └── settings.py
│
├── src/
│   ├── actuator_controller.py
│   ├── event_logger.py
│   ├── jetson_metrics.py
│   ├── threat_logic.py
│   └── ui.py
│
├── tools/
│   └── scripts de prueba y validación
│
├── models/
│   └── agroguard_yolov8n_v1.pt
│
├── data/
│   └── validation/
│
├── outputs/
│   └── agroguard_events.db
│
├── runs/
│   ├── events/
│   └── metrics/
│
└── notebooks/
    └── notebook de entrenamiento
```

## Programas principales

| Archivo | Función |
|---|---|
| `main.py` | Implementación base con procesamiento secuencial |
| `main2.py` | Variante de baja latencia con descarte de fotogramas atrasados |
| `main_metrics.py` | Variante instrumentada para medir rendimiento, latencia y recursos |

Los tres archivos corresponden a diferentes etapas del desarrollo y la validación. No representan tres sistemas independientes.

## Plataforma experimental

La implementación fue evaluada con:

- NVIDIA Jetson Orin Nano;
- JetPack 6.2.3;
- CUDA 12.6;
- Python 3.10;
- PyTorch con soporte CUDA;
- Ultralytics 8.4.117;
- NumPy 1.26.4;
- OpenCV 4.10.0;
- Jetson.GPIO;
- modelo YOLOv8n ajustado.

Las versiones forman parte de la configuración experimental. Cambios en PyTorch, CUDA, NumPy, OpenCV o Ultralytics pueden afectar la compatibilidad y el desempeño.

## Modelo de visión artificial

El modelo ajustado utilizado por el sistema se encuentra en:

```text
models/agroguard_yolov8n_v1.pt
```

El detector trabaja con las clases:

```text
person
boat
```

El conjunto de datos documentado contiene 997 imágenes:

| Subconjunto | Imágenes |
|---|---:|
| Entrenamiento | 760 |
| Validación | 127 |
| Prueba | 110 |
| Total | 997 |

El proceso de preparación, entrenamiento y evaluación del detector se documenta en el notebook de Google Colab incluido en el repositorio.

## Configuración

Los parámetros generales se encuentran en:

```text
config/settings.py
```

Entre ellos:

- nombre del modelo;
- umbral de confianza;
- tamaño de entrada;
- umbral de agrupación;
- tiempos de persistencia;
- tiempo de rearme;
- duración previa y posterior de las evidencias.

Las regiones de interés y configuraciones particulares de las escenas se encuentran en:

```text
config/scenes.py
```

Las coordenadas de las regiones de interés son específicas para las resoluciones y encuadres utilizados durante la validación.

## Instalación en NVIDIA Jetson

Se recomienda crear el entorno directamente en la Jetson. No debe copiarse un entorno virtual creado en Windows, debido a las diferencias entre las arquitecturas x86-64 y ARM64.

Ejemplo de creación del entorno:

```bash
python3 -m venv --system-site-packages ~/venvs/agroguard
source ~/venvs/agroguard/bin/activate
```

Antes de ejecutar AgroGuard, se debe comprobar el funcionamiento de PyTorch y CUDA:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('Disponible:', torch.cuda.is_available())"
```

La salida debe indicar:

```text
Disponible: True
```

También se deben comprobar las dependencias principales:

```bash
python -c "import cv2, numpy, ultralytics; print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__); print('Ultralytics:', ultralytics.__version__)"
```

> **Advertencia:** la versión de PyTorch debe ser compatible con JetPack, CUDA y la arquitectura ARM64 de la Jetson.

## Ejecución

Los comandos deben ejecutarse desde la raíz del proyecto y con el entorno virtual activado.

### Implementación base

```bash
python main.py --source data/validation/clips/camino/ROAD_GROUP_ZONE_001.mp4
```

### Variante de baja latencia

```bash
python main2.py --source data/validation/clips/camino/ROAD_GROUP_ZONE_001.mp4
```

### Variante con métricas

```bash
python main_metrics.py --source data/validation/clips/camino/ROAD_GROUP_ZONE_001.mp4
```

La fuente proporcionada debe contar con una configuración espacial correspondiente en `config/scenes.py`.

## Validación funcional

La validación se realizó mediante segmentos de video con resultados esperados para las reglas:

- `GROUP_DETECTED`;
- `RESTRICTED_ZONE_INTRUSION`;
- `BOAT_INTRUSION`.

Se evaluaron 46 segmentos y se obtuvo correspondencia entre la respuesta esperada y la respuesta generada en los 46 casos.

Este resultado debe interpretarse como:

> 100 % de concordancia funcional dentro del conjunto experimental evaluado.

No debe interpretarse como:

> 100 % de precisión universal del detector o del sistema.

Los segmentos proceden de un número menor de videos fuente, por lo que su independencia estadística es limitada.

## Métricas de ejecución

Los scripts de validación permiten registrar:

- FPS de la fuente;
- FPS efectivos de procesamiento;
- fotogramas procesados;
- fotogramas descartados;
- tiempo de inferencia;
- latencia entre condición y evento;
- latencia entre condición y actuación GPIO;
- uso de CPU;
- uso de GPU;
- utilización de RAM;
- temperatura máxima reportada.

Los resultados tabulares se encuentran en:

```text
data/validation/
runs/metrics/
```

Las métricas corresponden a la plataforma, configuración y videos utilizados durante la campaña experimental.

## Base de datos

Los eventos confirmados se almacenan en:

```text
outputs/agroguard_events.db
```

La tabla de eventos contiene los siguientes campos:

| Campo | Descripción |
|---|---|
| `id` | Identificador del evento |
| `created_at` | Fecha y hora de creación |
| `scene` | Escena o video procesado |
| `video_time` | Instante del evento dentro del video |
| `frame_number` | Número de fotograma asociado |
| `event_type` | Tipo de amenaza confirmada |
| `person_count` | Personas contabilizadas |
| `boat_count` | Embarcaciones contabilizadas |
| `evidence_path` | Ruta de la imagen de evidencia |
| `video_path` | Ruta del clip asociado |

La base permite relacionar la decisión del sistema con la evidencia generada durante la ejecución.

Las rutas almacenadas pueden corresponder al entorno donde se realizó la prueba y no necesariamente serán válidas después de descargar el repositorio en otra computadora.

## Evidencias

Cuando se confirma un evento, el sistema puede generar:

```text
runs/events/
├── imagen del evento
└── clip de evidencia
```

El clip incorpora fotogramas anteriores y posteriores a la confirmación de la amenaza, según los valores configurados mediante:

```python
PRE_EVENT_SECONDS
POST_EVENT_SECONDS
```

Las evidencias incluidas en el repositorio deben utilizarse únicamente con fines académicos y respetando la privacidad de las personas registradas.

## Actuación experimental

La plataforma experimental utiliza:

```text
Jetson GPIO
    ↓
Interfaz mediante MOSFET
    ↓
LED y buzzer
```

El pulsador permite silenciar una alarma activa. Después del silenciamiento, el sistema requiere un intervalo continuo sin amenazas antes de rearmarse.

Esta actuación verifica la continuidad funcional entre:

```text
percepción
    ↓
decisión
    ↓
registro
    ↓
salida física
```

No representa la validación de los actuadores industriales considerados en el diseño de campo.

## Limitaciones

- La validación se realizó principalmente mediante videos previamente registrados.
- Las regiones de interés son específicas para cada escena.
- No se realizó una campaña prolongada en una camaronera real.
- No se evaluaron completamente lluvia, niebla, oscuridad, ensuciamiento de la óptica o contraluz extremo.
- La actuación industrial mediante módulos ADAM, relés, reflector y sirena no formó parte de la plataforma experimental.
- El gabinete diseñado no fue sometido a certificación ambiental como conjunto integrado.
- La optimización mediante TensorRT permanece como trabajo futuro.

## Privacidad y uso de los datos

Los datos fueron utilizados con fines académicos. Las imágenes, videos y evidencias que contengan personas deben manejarse de acuerdo con las autorizaciones aplicables.

Las fuentes públicas utilizadas para construir el conjunto de entrenamiento conservan sus licencias originales. La inclusión del código o del notebook en este repositorio no modifica ni reemplaza las licencias de los datasets de terceros.

## Reproducibilidad

Para reproducir completamente los resultados se requiere:

1. una NVIDIA Jetson Orin Nano compatible;
2. un entorno de software equivalente;
3. el modelo ajustado;
4. los videos o clips de validación;
5. la configuración espacial correspondiente;
6. el script de validación utilizado;
7. acceso a `tegrastats` para medir los recursos de la plataforma.

Los resultados pueden variar si se modifica el hardware, la resolución, el modelo, las versiones de las dependencias o los videos evaluados.

## Autor

Braulio Chou Lou García  
Carrera de Ingeniería Mecatrónica  
Espol 
Ecuador

## Estado del proyecto

Proyecto en fase de validación y documentación de resultados.

## Licencia

La licencia del código y del modelo se encuentra pendiente de definición según las condiciones de la institución académica, la titularidad del proyecto y los acuerdos con la organización colaboradora.
