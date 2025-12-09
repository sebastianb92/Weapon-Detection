# Detección de Armas: Pistolas y Cuchillos (YOLO/Ultralytics) 

## Descripción del Proyecto

Este proyecto de **Visión por Computador** se enfoca en la detección y localización automática de **pistolas** y **cuchillos** en imágenes, con aplicaciones directas en sistemas de videovigilancia inteligente y seguridad pública.

El sistema se basa en el *framework* **Ultralytics** para entrenar un modelo de la familia **YOLO** (You Only Look Once), optimizado para la detección de objetos en tiempo real.

---

## Objetivo del Proyecto

Desarrollar un modelo de detección de objetos robusto y de alto rendimiento, entrenado específicamente para identificar las clases `pistol` (pistola) y `knife` (cuchillo) en diversos escenarios.

---

## Integrantes

Este proyecto fue desarrollado como parte del **Proyecto de Innovación II** de la Maestría en Inteligencia Artificial.

* **Johan Sebastian Bonilla**
* **Edwin Gómez**

---

## Tecnologías y Herramientas

* **Modelo de Detección:** YOLO (Ultralytics)
* **Lenguaje:** Python
* **Framework de Entrenamiento:** Ultralytics
* **Librerías Clave:** `numpy`, `pandas`, `opencv-python` (`cv2`), `Pillow` (`PIL`), `matplotlib` (para análisis y pre-procesamiento).

---

## Dataset

El modelo fue entrenado con un conjunto de datos especializado en detección de armas, proporcionado por el **DaSCI (Instituto Andaluz Interuniversitario en Data Science and Computational Intelligence)**.

* **Fuente Principal:** [DaSCI Open Data - Detección de Armas](https://dasci.es/opendata/deteccion-de-armas-open-data/)
* **Subconjuntos utilizados:**
    * Detección de Pistolas (Pistol Detection)
    * Detección de Armas Blancas (Knife Detection)
    * Sohas Weapon Detection
* **Formato de Anotación:** PASCAL VOC (`.xml`), luego convertido a formato YOLO (`.txt`).
* **Clases de Detección:** `pistol` y `knife`
* **Total de Registros de *Bounding Boxes*:** 7436

---

## Métricas de Rendimiento

El entrenamiento del modelo final arrojó las siguientes métricas de desempeño sobre el conjunto de validación:

| Métrica | Definición | Valor |
| :--- | :--- | :--- |
| **mAP50** | *mean Average Precision* con IoU del 50% | **0.954** |
| **mAP50-95** | *mean Average Precision* promedio sobre umbrales IoU de 0.50 a 0.95 | **0.704** |
| **Precision** | | **0.928** |
| **Recall** | | **0.899** |

