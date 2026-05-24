# Sistema de Detección Automática de Armas en Imágenes y Video Mediante YOLO26s

Este proyecto hace parte del curso **Proyecto III de Innovación Tecnológica** de la Maestría en Inteligencia Artificial Aplicada (MIAA), Universidad Icesi, Cali, Colombia.

> **Estado del proyecto:** `Completado`

---

## Integrantes

| Rol | Nombre | GitHub | Correo |
|-----|--------|--------|--------|
| Líder del proyecto | Johan Sebastian Bonilla | [@sebastianb92](https://github.com/sebastianb92) | johan.bonilla@u.icesi.edu.co |
| Integrante | Edwin Felipe Gómez | [@edwingomez](https://github.com/edwingomez) | edgo1688@gmail.com |
| Instructor | Yesid Ospitia Medina | — | — |

---

## Contacto

Para preguntas, sugerencias o interés en contribuir, puedes contactar al líder del proyecto a través de GitHub o al correo indicado en la tabla anterior.

---

## Descripción general del proyecto

El objetivo de este proyecto es desarrollar un prototipo funcional *end-to-end* para la detección automática de armas (pistolas y cuchillos) en imágenes y video, orientado a fortalecer los sistemas de videovigilancia inteligente en contextos de seguridad pública colombiana. El sistema emplea la arquitectura **YOLO26s** entrenada con transfer learning sobre un dataset consolidado de 7.436 anotaciones (DaSCI Open Data), e integra un módulo de **inteligencia artificial generativa multimodal** (LLaMA 4 Vision vía Groq) para generar descripciones automáticas de los eventos detectados. Todos los componentes se presentan mediante una **interfaz interactiva desarrollada con Gradio**.

El proyecto demuestra la viabilidad técnica del sistema, alcanzando mAP@0.5 = 0.954 en validación, y documenta con transparencia la brecha de *domain shift* (~30 pp de caída) observada en pruebas con imágenes reales de contexto colombiano.

---

## Métodos utilizados

- Detección de objetos en tiempo real (single-stage, anchor-free)
- Transfer learning y fine-tuning sobre modelos preentrenados en COCO
- Análisis exploratorio de datos (EDA) y auditoría de calidad de datasets
- Balanceo de clases mediante oversampling
- Aumentación de datos (geométrica, fotométrica, simulación CCTV)
- Evaluación cuantitativa con métricas estándar (mAP, Precisión, Recall, IoU)
- Evaluación cualitativa en dominio real (context shift analysis)
- Inferencia en imágenes individuales y video frame a frame
- Generación de lenguaje natural con modelos de visión-lenguaje (VLM)

---

## Tecnologías y herramientas

| Categoría | Herramientas |
|-----------|-------------|
| Lenguaje | Python 3.10+ |
| Detección de objetos | Ultralytics YOLO26s |
| Aumentación de datos | Albumentations |
| Procesamiento de video | OpenCV |
| IA Generativa | LLaMA 4 Scout 17B (Meta AI) vía Groq API |
| Interfaz interactiva | Gradio |
| Entorno de desarrollo | Google Colab (GPU Tesla T4) |
| Gestión de datos | Pandas, NumPy |
| Visualización | Matplotlib, Seaborn |
| Formato de anotaciones | Pascal VOC (XML) → YOLO (TXT) |

---

## Descripción detallada

### Fuentes de datos

El dataset utilizado proviene de la iniciativa **DaSCI Open Data** (Universidad de Granada, España), compuesto por tres subdatasets consolidados:

| Subdataset | Imágenes aprox. | Descripción |
|------------|-----------------|-------------|
| `Knife_detection` | ~2.078 | Imágenes de cuchillos en diversos contextos |
| `Pistol_detection` | ~3.000 | Imágenes de pistolas y revólveres |
| `Sohas_weapon-Detection` | ~2.358 | Muestras adicionales mixtas |

- **Total de anotaciones:** 7.436 (7.435 válidas tras limpieza)
- **Clases:** 2 — `knife` (0), `pistol` (1)
- **Formato original:** Pascal VOC (XML, coordenadas absolutas)
- **Distribución:** Pistol ~67% / Knife ~33% — ratio de desbalance 2.1:1

### Preguntas e hipótesis

**Pregunta de investigación:** ¿Es posible desarrollar un sistema de detección automática de armas (pistolas y cuchillos) en imágenes y video que alcance un desempeño suficiente para ser integrado como componente de apoyo en sistemas de videovigilancia colombianos, y que proporcione descripciones automáticas del evento detectado mediante IA generativa?

**Hipótesis (H1):** El entrenamiento de YOLO26s mediante transfer learning sobre el dataset consolidado, con balanceo de clases y aumentación orientada a vigilancia, permite alcanzar mAP@0.5 ≥ 0.90 en validación. Sin embargo, se anticipa una brecha de *domain shift* respecto a escenarios reales colombianos que limitará la efectividad operativa directa sin adaptaciones adicionales.

### Resultados principales

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| mAP@0.5 | **0.954** | Supera la hipótesis (≥ 0.90) ✓ |
| mAP@0.5:0.95 | 0.706 | Desempeño robusto en localización estricta |
| Precisión | 0.928 | 7.2% de falsas alarmas |
| Recall | 0.901 | 9.9% de armas no detectadas |
| Recall en contexto real colombiano | ~60% | Brecha de domain shift de ~30 pp |

### Desafíos y bloqueos identificados

- **Domain shift:** brecha de ~30 pp entre validación controlada y contexto real colombiano, atribuible a la escasez de imágenes nocturnas, desenfoque, fondos complejos y sesgo geográfico del dataset (imágenes europeas/norteamericanas).
- **Desbalance de clases:** ratio inicial 2.1:1 (pistola:cuchillo), mitigado con oversampling.
- **Evaluación del módulo GenAI:** evaluado cualitativamente; pendiente evaluación formal con métricas NLG (BLEU, ROUGE, BERTScore).
- **Recursos computacionales:** entrenamiento limitado a 25 épocas con batch=16 por restricciones de GPU en Google Colab.

---

## Cómo empezar

### Requisitos previos

```bash
# Clonar el repositorio
git clone https://github.com/sebastianb92/ecomarket-rag-solution.git
cd ecomarket-rag-solution

# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias principales

```
ultralytics>=8.0.0
albumentations>=1.3.0
gradio>=4.0.0
groq>=0.4.0
opencv-python>=4.8.0
pandas>=2.0.0
matplotlib>=3.7.0
```

### Datos

Los datos crudos en formato Pascal VOC se obtienen desde [DaSCI Open Data](https://dasci.es/opendata/deteccion-de-armas-open-data/) (Universidad de Granada). Una vez descargados, colócalos en la carpeta `data/raw/` con la siguiente estructura:

```
data/
├── raw/
│   ├── Knife_detection/
│   │   ├── images/
│   │   └── annotations/
│   ├── Pistol_detection/
│   │   ├── images/
│   │   └── annotations/
│   └── Sohas_weapon-Detection/
│       ├── images/
│       └── annotations/
├── processed/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── val/
│       ├── images/
│       └── labels/
└── dataset.yaml
```

### Ejecución paso a paso

```bash
# Paso 1: EDA, auditoría de gaps, preprocesamiento y entrenamiento
jupyter notebook notebooks/01_eda.ipynb

# Paso 2: Inferencia en imagen y video + descripción GenAI
jupyter notebook notebooks/02_inference.ipynb

# Paso 3: Aplicación interactiva Gradio
jupyter notebook notebooks/03_app.ipynb
```

> **Nota:** Para ejecutar el módulo GenAI necesitas una API key de Groq. Créala en [console.groq.com](https://console.groq.com) y agrégala como variable de entorno:
> ```bash
> export GROQ_API_KEY="tu_api_key_aqui"
> ```
> En Google Colab puedes usar `userdata.get('GROQ_API_KEY')` desde los Secrets del entorno.

### Pesos del modelo

El modelo entrenado (`best.pt`) está disponible en la carpeta `models/`. Si deseas reentrenar desde cero, ejecuta el notebook `01_eda.ipynb` completo.

---

## Estructura del repositorio

```
├── data/
│   ├── raw/                  # Datos originales (no modificar)
│   └── processed/            # Datos preparados para entrenamiento
├── models/
│   └── best.pt               # Pesos del modelo YOLO26s entrenado
├── notebooks/
│   ├── 01_eda.ipynb          # EDA, auditoría, preprocesamiento y entrenamiento
│   ├── 02_inference.ipynb    # Inferencia en imagen/video + módulo GenAI
│   └── 03_app.ipynb          # Aplicación interactiva Gradio
├── results/
│   ├── metrics/              # Curvas de entrenamiento, matrices de confusión
│   └── inference/            # Imágenes y videos procesados de ejemplo
├── requirements.txt
├── dataset.yaml              # Configuración del dataset para Ultralytics
└── README.md
```

---

## Entregables principales

| Entregable | Descripción | Enlace |
|------------|-------------|--------|
| `01_eda.ipynb` | EDA completo, auditoría de gaps, preprocesamiento y entrenamiento YOLO26s | [Ver notebook](notebooks/01_eda.ipynb) |
| `02_inference.ipynb` | Pipeline de inferencia en imagen y video + descripción automática con GenAI | [Ver notebook](notebooks/02_inference.ipynb) |
| `03_app.ipynb` | Aplicación interactiva Gradio (3 pestañas: imagen, video, métricas) | [Ver notebook](notebooks/03_app.ipynb) |
| `best.pt` | Pesos del modelo YOLO26s entrenado (mAP@0.5 = 0.954) | [Ver modelo](models/best.pt) |
| Documento del proyecto | Informe técnico completo en formato IEEE (Icesi) | [Ver PDF](docs/Proyecto_MIAA_Deteccion_Armas.pdf) |

---

## Consideraciones éticas y legales

El sistema está diseñado exclusivamente como **herramienta de apoyo** con supervisión humana obligatoria. Su desarrollo y eventual despliegue deben cumplir con:

- **Ley 1581 de 2012** — Protección de datos personales en Colombia
- **Resolución 2852 de 2006** — Regulación de sistemas CCTV en Colombia
- **Lineamientos UNESCO (2023)** y **OCDE (2021)** sobre ética en IA
- **AI Act (UE, 2024)** como referencia comparada para sistemas de alto riesgo

Cualquier despliegue operativo requiere evaluación de impacto en protección de datos, protocolos de gobernanza y mecanismos para prevenir el perfilamiento discriminatorio.

---

## Licencia

Este proyecto fue desarrollado con fines académicos en el marco de la Maestría en Inteligencia Artificial Aplicada, Universidad Icesi. Los datos utilizados provienen de DaSCI Open Data (Universidad de Granada) bajo licencia de acceso libre para investigación.