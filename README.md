# Car Counter - Contador de Vehículos

Sistema de detección, tracking y conteo de vehículos usando YOLOv8 y ByteTrack.

## 🚀 Características

- **Detección de vehículos** usando YOLOv8 (car, motorcycle, bus, truck)
- **Tracking de objetos** con ByteTrack para seguir vehículos entre frames
- **Conteo de vehículos** usando línea de conteo configurable
- **Anotación de video** con bounding boxes, IDs de tracker y contadores

## 📋 Requisitos

- Python 3.12+
- CUDA (opcional, para aceleración GPU)
- `uv` (gestor de paquetes Python) o `pip`

## 🔧 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/IgnacioFucksmann/CarCounter.git
cd CarCounter
```

### 2. Crear entorno virtual

```bash
# Con uv (recomendado)
uv venv

# O con venv estándar
python -m venv .venv
```

### 3. Activar entorno virtual

```bash
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate  # Windows
```

### 4. Instalar dependencias

```bash
# Con uv
uv pip install -r requirements.txt

# O con pip
pip install -r requirements.txt
```

### 5. Instalar ByteTrack

ByteTrack no está disponible en PyPI, debe instalarse desde GitHub:

```bash
bash install_bytetrack.sh
```

Este script:
- Clona ByteTrack desde GitHub
- Corrige versiones de dependencias
- Instala ByteTrack en modo desarrollo

**Nota:** Si prefieres instalarlo manualmente, consulta `install_bytetrack.sh` para ver los pasos.

## 🎯 Uso

### Configuración básica

Edita `main.py` para configurar:

```python
input_video = "data/vehicle-counting.mp4"  # Video de entrada
output_video = "output/vehicle-counting-result.mp4"  # Video de salida
model_name = "yolov8x.pt"  # Modelo YOLOv8 (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
line_start = Point(50, 1500)  # Punto inicial de línea de conteo
line_end = Point(3840 - 50, 1500)  # Punto final de línea de conteo
```

### Ejecutar

```bash
python main.py
```

El script:
1. Carga el modelo YOLOv8 (se descarga automáticamente la primera vez)
2. Procesa el video frame por frame
3. Detecta y rastrea vehículos
4. Cuenta vehículos que cruzan la línea
5. Guarda el video procesado con anotaciones
6. Muestra estadísticas finales

### Salida

El video procesado se guarda en `output/vehicle-counting-result.mp4` con:
- Bounding boxes alrededor de cada vehículo
- Labels con ID de tracker, clase y confianza
- Línea de conteo con contador de vehículos

Al finalizar, se muestran las estadísticas:
```
Vehicles counted (in):  45
Vehicles counted (out): 42
Total:                  87
```

## 📁 Estructura del Proyecto

```
car_counter/
├── data/                    # Videos de entrada
│   └── vehicle-counting.mp4
├── output/                  # Videos procesados (ignorado por Git)
├── utils/                   # Utilidades
│   ├── tracking_utils.py    # Funciones de tracking
│   └── TRACKING_UTILS_EXPLICACION.md
├── libs/                    # Dependencias externas
│   └── ByteTrack/           # ByteTrack (ignorado por Git)
├── model.py                 # Clase VehicleDetector
├── processor.py             # Clase VideoProcessor
├── main.py                  # Script principal
├── install_bytetrack.sh     # Script de instalación de ByteTrack
├── requirements.txt         # Dependencias Python
└── README.md                # Este archivo
```

## 🔍 Modelos YOLOv8 Disponibles

- `yolov8n.pt` - Nano (más rápido, menos preciso)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - XLarge (más preciso, más lento) - **Por defecto**

Los modelos se descargan automáticamente la primera vez que se usan.

## ⚙️ Configuración Avanzada

### Cambiar clases detectadas

En `model.py`, modifica `class_ids`:

```python
detector = VehicleDetector(
    model_name="yolov8x.pt",
    class_ids=[2, 3, 5, 7]  # car, motorcycle, bus, truck
)
```

### Ajustar parámetros de ByteTrack

En `processor.py`, modifica `BYTETrackerArgs`:

```python
@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25      # Umbral de confianza
    track_buffer: int = 30           # Frames de buffer
    match_thresh: float = 0.8       # Umbral de matching
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
```

## 🐛 Solución de Problemas

### Error: "No module named 'yolox'"

ByteTrack no está instalado. Ejecuta:
```bash
bash install_bytetrack.sh
```

### Error: "No module named 'torch'"

Instala PyTorch:
```bash
# Con CUDA 12.6
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# O solo CPU
uv pip install torch torchvision
```

### El video se procesa muy lento

- Usa un modelo más pequeño: `yolov8n.pt` o `yolov8s.pt`
- Asegúrate de tener GPU con CUDA instalado
- Reduce la resolución del video de entrada

## 📚 Documentación Adicional
- `install_bytetrack.sh` - Script comentado con explicación de cada paso

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8
- [FoundationVision](https://github.com/FoundationVision/ByteTrack) - ByteTrack
- [Roboflow Supervision](https://github.com/roboflow/supervision) - Utilidades de video

## 📧 Contacto

Para preguntas o sugerencias, abre un issue en el repositorio.


