# RESUMEN Portafolio de anális
## Indicadores de Privacidad, Anonimización y Control de Acceso a Datos
---

## Síntesis General
| # | Indicador | Estado | Cumplimiento |
|---|-----------|--------|--------------|
| 1 | Trabaja con grandes volúmenes de datos | **SE CUMPLE** | 1.5M registros BD + 21 videos + miles de esqueletos |
| 2 | Verifica anonimización de datos | **SE CUMPLE** | UUIDs + keypoints sin rostro + normalización + CLIP |
| 3 | Especifica procesos y estándares | **SE CUMPLE** | Auth/Autorización + filtrado espacial + validación confianza |
| 4 | Presenta registros de auditoría | **SE CUMPLE** | Logs + CSVs + manifests con trazabilidad completa |

---

## Hallazgos Principales

### INDICADOR 1: Volúmenes de Datos Masivos

**Donde se cumple:**

- **Base de Datos PostGIS:** 1,592,490 registros de personas en el parque infantil
- **Almacenamiento Azure Blob:** 21 videos de múltiples cámaras
- **Esqueletos procesados:** Miles de puntos articulares (17 por persona)
- **Grafos generados:** Matrices de adyacencia panorámica

**Archivos clave:**
```
✓ reading-db.ipynb (BD PostGIS con ST_Intersects)
✓ Playground/data/videos.csv (21 videos catalogados)
✓ Playground/data/npy/*.npy (esqueletos normalizados)
✓ Playground/data/graph/ (grafos con objetos)
```

---

### INDICADOR 2: Anonimización de Datos

**Mecanismos implementados:**

1. **UUIDs en Base de Datos**
   - Campo `id_person`: UUID v4 aleatorio (ej: `b17e1ad8-0c2c-44e4-ba72-54c2dd3881a4`)
   - Ningún nombre, rostro o dato biométrico identificable

2. **Extracción de Esqueletos sin Rostro**
   - YOLOv8 para detección de cuerpo (no rostro)
   - MediaPipe Pose: 17 keypoints de articulaciones
   - **Sin información facial**

3. **Normalización de Postura**
   - Traslación: pelvis → origen (0,0)
   - Escala: torso normalizado a 1
   - Resultado: coordenadas relativas [0,1]

4. **Almacenamiento Anonimizado**
   - Arrays NumPy: solo valores numéricos
   - Shape: (T=48, K_max=4, 17 articulaciones, 2 coords)
   - **Sin metadatos identificables**

5. **Etiquetado Automático CLIP**
   - Clasifica comportamiento desde esqueletos
   - No tiene acceso a rostros o identidades
   - Etiquetas: Transit, Social, Play Normal/Risk, Adult, Negative Contact

**Archivos clave:**
```
✓ Playground/scripts/extract_skeletons.py (normalización)
✓ Playground/data/npy/*.npy (esqueletos anonimizados)
✓ Playground/scripts/label_with_clip.py (etiquetado anónimo)
```

---

### INDICADOR 3: Procesos y Estándares de Validación

**Estándares implementados:**

| Proceso | Estándar | Validación |
|---------|----------|-----------|
| **Autenticación BD** | User + IP privada (40.84.231.179:5434) | VNet exclusiva Azure |
| **Autorización Azure** | SAS Token con expiración (2026-04-02) | HTTPS obligatorio |
| **Filtrado Espacial** | ST_Intersects con playgroundROI.gpkg | Solo datos dentro del parque |
| **Normalización** | Función determinística `normalize_skeleton()` | Pelvis=origen, torso=1 |
| **Confianza de Etiquetas** | Umbral `--min-conf 0.6` | Descarta baja calidad |
| **Reproducibilidad** | `random_state=42` | Splits consistentes |
| **Validación de Grafos** | Config por cámara (objects.yaml) | Centroides normalizados [0,1] |

**Archivos clave:**
```
✓ reading-db.ipynb (auth + autorización)
✓ playgroundROI.gpkg (filtrado espacial)
✓ extract_skeletons.py (normalización)
✓ filter_and_split_labels.py (confianza + reproducibilidad)
✓ build_panoramic_graph.py (validación grafos)
```

---

### INDICADOR 4: Registros de Auditoría

**Registros implementados:**

1. **Conexión Base de Datos**
   - Ubicación: `reading-db.ipynb`
   - Salida: "Total dentro del ROI: 1,592,490"
   - Evidencia: Query documentada con parámetros

2. **Descargas de Azure**
   - Ubicación: `reading-db.ipynb`
   - Salida: "Downloading {blob_path}"
   - Evidencia: Progress bar + destino documentado

3. **Extracción de Esqueletos**
   - Ubicación: `extract_skeletons.py`
   - Salida: Logging INFO en consola
   - Evidencia: Archivos .npy generados con nombre único

4. **Etiquetado Automático**
   - Archivo: `vlm_labels_clip.csv`
   - Contenido: file, label, conf_weight
   - Evidencia: 3 columnas de trazabilidad

5. **Filtrado por Confianza**
   - Archivo: `vlm_labels_filtered_0.60.csv`
   - Contenido: Todas las muestras ≥ conf
   - Evidencia: Logs de entrada/salida

6. **Exclusiones de Baja Confianza**
   - Archivo: `vlm_labels_lowconf_below_0.60.csv`
   - Contenido: Muestras rechazadas
   - Evidencia: Disponibles para auditoría

7. **Split Entrenamiento/Validación**
   - Archivos: `train.csv` (80%), `val.csv` (20%)
   - Contenido: Stratified por clase
   - Evidencia: Reproducible con `random_state=42`

**Archivos clave:**
```
✓ reading-db.ipynb (conexión + descargas)
✓ extract_skeletons.py (extracción)
✓ label_with_clip.py (etiquetado)
✓ filter_and_split_labels.py (filtrado)
✓ Playground/data/graph/with_objects/ (CSVs de auditoría)
```

---
## Matriz de Control de Acceso

```
NIVEL 1: AUTENTICACIÓN
├─ PostGIS: user/password + IP whitelist
├─ Azure: SAS Token (limitado, con expiración)
└─ Sistema de Archivos: Permisos OS

NIVEL 2: AUTORIZACIÓN
├─ PostGIS: Usuario admin con lectura
├─ Azure: Token solo para contenedor 'crowdcounting'
└─ Sistema: Grupo de proyecto

NIVEL 3: ANONIMIZACIÓN
├─ BD: UUIDs en lugar de nombres
├─ Keypoints: Solo articulaciones, sin rostro
├─ Normalización: Coordenadas relativas [0,1]
└─ CLIP: Clasifica comportamiento, no identidad

NIVEL 4: AUDITORÍA
├─ Logs de conexión
├─ Manifests de descargas
├─ CSVs de etiquetado
├─ Registros de filtrado
└─ Splits reproducibles
```

---
### 1. **INDICADORES_PRIVACIDAD_Y_ANONIMIZACION.md**
   - Descripción detallada de cada indicador
   - Ubicaciones exactas en el código
   - Implementación técnica
   - Ejemplos de código
   - 480+ líneas de documentación

### 2. **MATRIZ_TRAZABILIDAD_PRIVACIDAD.md**
   - Tabla de componentes vs. volúmenes
   - Matriz de anonimización por capa
   - Estándares y validaciones
   - Tabla de registros de auditoría
   - Matriz de control de acceso visual
---

## Conclusiones

### Cumplimiento Integral

El proyecto cumple con los siguientes indicadores:

1. **Grandes volúmenes:** 1.5M registros + 21 videos + miles de esqueletos
2. **Anonimización:** UUIDs + keypoints sin rostro + normalización + CLIP
3. **Estándares:** Auth, autorización, filtrado, validación, reproducibilidad
4. **Auditoría:** Logs, CSVs, manifests con trazabilidad completa

---

## Ubicación de Archivos

Ambos documentos están disponibles en:
```
c:\Users\edosa\Documents\Uni\7to\Reto\R2\Socio-Formador-IA-Avanzada\
├─ INDICADORES_PRIVACIDAD_Y_ANONIMIZACION.md (documentación completa)
└─ JUSTIFIACION_PRIVACIDAD_DATOS.md (Razón de vulnerabilidades)
```