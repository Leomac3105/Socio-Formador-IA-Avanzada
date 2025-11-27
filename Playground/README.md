# Pipeline Playground

Este directorio contiene todo lo necesario para crear los insumos panorámicos que pide el reto (ventanas `[T,K_max,17,2]`, `data/videos.csv`, `configs/objects.yaml`, etc.).

## Estructura

- `data/videos.csv` – listado de ≥100 escenas filtradas (video_id, cámara, timestamps, blob_path, n_detections).
- `data/npy/` – se guardará un `.npy` por ventana `[48,4,17,2]` normalizada.
- `Downloads/` – coloca aquí los videos descargados desde blob/storage. El script toma el `blob_path` y lo busca en este directorio.
- `configs/objects.yaml` – centroides (0..1) de objetos por cámara. Copiamos la versión del equipo 1 para tener un punto de partida; actualízala según tus anotaciones.
- `scripts/extract_skeletons.py` – script CLI que ejecuta detección (YOLOv8), un tracker simple, MediaPipe Pose y genera las ventanas.
- `yolov8n.pt` – pesos livianos para detección de personas (puedes reemplazar por modelo propio).
- `extract_skeletons.ipynb` / `filtradoVideos.ipynb` – notebooks exploratorios (selección y pruebas).

## Instalación rápida

```bash
cd equipo2/Socio-Formador-IA-Avanzada/Playground
conda create -n playground python=3.10 -y
conda activate playground
pip install ultralytics mediapipe opencv-python pandas numpy azure-storage-blob
```

> Si tienes GPU Nvidia instala también `torch` + `torchvision` acordes a tu CUDA.

## Descargar videos desde Azure Blob Storage

1. Consigue una `connection string` o `AZURE_STORAGE_ACCOUNT_KEY` con acceso al contenedor `videos` de la cuenta `socioformadorstorage`.
2. Exporta la credencial (elige una de las dos opciones):
   ```bash
   export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=...=="
   # o
   export AZURE_STORAGE_ACCOUNT_KEY="XXXX"
   ```
3. Ejecuta el script de descarga:
   ```bash
   python scripts/download_videos.py \
     --videos-csv data/videos.csv \
     --output-dir Downloads
   ```
   Cada fila de `videos.csv` se descargará a `Downloads/<blob_name>`. El script salta archivos ya presentes, así que puedes relanzarlo cuando agregues nuevas escenas.

## Generar esqueletos normalizados

1. Copia los videos filtrados en `Downloads/` (mismos nombres que `blob_path`).
2. Verifica/edita `data/videos.csv`.
3. Ejecuta:

```bash
python scripts/extract_skeletons.py \
  --videos-csv data/videos.csv \
  --downloads-dir Downloads \
  --output-dir data/npy \
  --yolo-weights yolov8n.pt \
  --window 48 \
  --k-max 4
```

El script:

1. Muestrea cada video a ~12 FPS.
2. Detecta personas con YOLOv8 (clase 0).
3. Aplica MediaPipe Pose sobre cada recorte, normaliza a pelvis-origen / torso=1.
4. Agrupa resultados en ventanas deslizantes (`stride=T/2`) y guarda cada tensor como `videoid_winnnn.npy`.

## Próximos entregables

- `data/npy/*.npy` – listo para alimentar el *feeder*.
- `configs/objects.yaml` verificada (centroides por cámara).
- Script `build_panoramic_graph` (pendiente) para expandir `V'` y generar `J/B/JM/BM`.
- Evidencia de un forward del feeder usando los tensores generados.
