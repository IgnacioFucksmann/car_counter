#!/bin/bash
# ============================================================================
# Script para instalar ByteTrack desde GitHub
# ByteTrack es el tracker estándar y más efectivo para tracking de vehículos
# 
# Este script automatiza la instalación manual de ByteTrack.
# Para entender cada paso, consulta INSTALACION_BYTETRACK.md
# ============================================================================

# set -e: Si cualquier comando falla, el script se detiene inmediatamente
# Esto evita continuar con pasos siguientes si algo sale mal
set -e

echo "🚀 Iniciando instalación de ByteTrack desde GitHub..."
echo ""

# ============================================================================
# PASO 1: Instalar PyTorch con soporte CUDA (requerido por ByteTrack)
# ============================================================================
# ByteTrack necesita torch durante la compilación, así que lo instalamos primero
# Usamos CUDA 12.6 (cu126) para mejor rendimiento con GPUs modernas
# Si no tienes GPU, puedes cambiar a CPU más abajo
# IMPORTANTE: Esto se instala en el entorno virtual actual, no en el directorio temporal
echo "🔥 Instalando PyTorch con soporte CUDA 12.6..."
if command -v uv &> /dev/null; then
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
fi
echo "✅ PyTorch instalado"
echo ""

# ============================================================================
# PASO 2: Crear directorio temporal
# ============================================================================
# mktemp -d crea un directorio temporal único
# Ejemplo: /tmp/tmp.XXXXXXXXXX
# Usamos un directorio temporal para no contaminar el proyecto actual
TEMP_DIR=$(mktemp -d)
echo "📁 Directorio temporal creado: $TEMP_DIR"
cd "$TEMP_DIR"

# ============================================================================
# PASO 3: Clonar ByteTrack desde GitHub
# ============================================================================
# git clone descarga todo el código fuente de ByteTrack
# Esto es necesario porque ByteTrack no está en PyPI
# Usamos el repositorio oficial: FoundationVision/ByteTrack
echo "📥 Clonando ByteTrack desde GitHub (FoundationVision/ByteTrack)..."
git clone https://github.com/FoundationVision/ByteTrack.git
cd ByteTrack
echo "✅ ByteTrack clonado correctamente"
echo ""

# ============================================================================
# PASO 4: Corregir versiones antiguas de ONNX y ONNXRuntime (Workaround)
# ============================================================================
# sed -i: Edita el archivo in-place (modifica el archivo directamente)
# Reemplaza versiones antiguas por versiones modernas con wheels precompilados
# || true: Si el comando falla (porque ya está corregido), continúa sin error
# 
# ¿Por qué?
# - onnx 1.8.1 y 1.9.0 requieren cmake para compilar desde fuente
# - onnxruntime 1.8.0 ya no está disponible en PyPI (versiones mínimas: 1.17.0+)
# - Las versiones modernas tienen wheels precompilados, evitando compilación
# - Esto hace la instalación más rápida y sin dependencias del sistema
echo "🔧 Aplicando workaround: actualizando ONNX y ONNXRuntime a versiones modernas..."
# Actualizar onnx
sed -i 's/onnx==1.8.1/onnx>=1.12.0/g' requirements.txt || true
sed -i 's/onnx==1.9.0/onnx>=1.12.0/g' requirements.txt || true
# Actualizar onnxruntime (versión 1.8.0 ya no existe, usar >=1.17.0)
sed -i 's/onnxruntime==1.8.0/onnxruntime>=1.17.0/g' requirements.txt || true
sed -i 's/onnxruntime==1.8.1/onnxruntime>=1.17.0/g' requirements.txt || true
echo "✅ requirements.txt actualizado"
echo ""

# ============================================================================
# PASO 5: Instalar dependencias de ByteTrack
# ============================================================================
# pip install -r requirements.txt lee el archivo requirements.txt
# e instala todas las librerías Python que ByteTrack necesita
# -q: modo silencioso (quiet), menos output en pantalla
# Usamos uv pip si está disponible, sino pip normal
echo "📦 Instalando dependencias de ByteTrack..."
if command -v uv &> /dev/null; then
    uv pip install -q -r requirements.txt
else
    pip install -q -r requirements.txt
fi
echo "✅ Dependencias instaladas"
echo ""

# ============================================================================
# PASO 6: Instalar ByteTrack en modo desarrollo
# ============================================================================
# pip install -e . (o uv pip install -e .):
#   - Compila el código C/C++ de ByteTrack
#   - Instala ByteTrack en modo "desarrollo" (editable)
#   - Crea enlaces simbólicos al código fuente
# 
# Modo desarrollo significa que si modificas el código,
# los cambios se reflejan automáticamente sin reinstalar
# 
# --no-build-isolation: Usa las dependencias del entorno actual (incluyendo torch)
# Usamos uv pip si está disponible, sino pip normal
echo "🔨 Compilando e instalando ByteTrack (esto puede tardar unos minutos)..."
if command -v uv &> /dev/null; then
    uv pip install -e . --no-build-isolation --quiet
else
    pip install -e . --no-build-isolation --quiet
fi
echo "✅ ByteTrack compilado e instalado"
echo ""

# ============================================================================
# PASO 7: Instalar dependencias adicionales necesarias
# ============================================================================
# Estas librerías no están en requirements.txt de ByteTrack
# pero son necesarias para que funcione correctamente con YOLOv8:
#   - cython_bbox: Cálculo rápido de intersecciones de bounding boxes
#   - onemetric: Funciones de métricas de tracking (IoU, etc.)
#   - loguru: Sistema de logging mejorado
#   - lap: Algoritmo húngaro para matching de objetos
#   - thop: Cálculo de operaciones FLOPs
echo "📦 Instalando dependencias adicionales..."
if command -v uv &> /dev/null; then
    uv pip install -q cython_bbox onemetric loguru lap thop
else
    pip install -q cython_bbox onemetric loguru lap thop
fi
echo "✅ Dependencias adicionales instaladas"
echo ""

# ============================================================================
# PASO 8: Limpiar archivos temporales
# ============================================================================
# Volvemos al directorio original
cd -
# rm -rf: Elimina recursivamente el directorio temporal
# Esto limpia los archivos descargados que ya no necesitamos
echo "🧹 Limpiando archivos temporales..."
rm -rf "$TEMP_DIR"
echo "✅ Limpieza completada"
echo ""

# ============================================================================
# Verificación final
# ============================================================================
echo "✅ ByteTrack instalado correctamente!"
echo ""
echo "Para verificar la instalación, ejecuta en Python:"
echo "  import sys"
echo "  sys.path.append('ByteTrack')"
echo "  from yolox.tracker.byte_tracker import BYTETracker"
echo "  print('✅ ByteTrack funciona correctamente')"
echo ""

