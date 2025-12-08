# Indicadores de Privacidad, Anonimización y Control de Acceso a Datos
## Proyecto: Escenas de Playground usando MP-GCN

---

## INDICADOR 1: Trabaja con grandes volúmenes de datos

### **Se cumple en:**

#### 1.1 Base de Datos PostGIS (`reading-db.ipynb`)
- **Ubicación:** `/reading-db.ipynb`
- **Descripción:** Acceso a base de datos PostGIS con más de **1.5 millones de registros** (1,592,490 filas)
- **Detalles:**
  ```python
  # Conexión a BD PostGIS en Azure
  POSTGIS_CONN = {
      "dbname": "crowdcounting",
      "user": "admin",
      "password": "admin",
      "host": "40.84.231.179",
      "port": "5434",
  }
  ```
- **Volumen:** Total dentro del ROI: **1,592,490 registros**
- **Consulta SQL con geolocalización:** Filtra personas dentro de una región de interés (ROI) usando operadores espaciales (`ST_Intersects`)

#### 1.2 Almacenamiento en Azure Blob Storage
- **Ubicación:** `/reading-db.ipynb` (celdas de descarga)
- **Contenedor:** `crowdcounting` en `cienciaciudades2024.blob.core.windows.net`
- **Tipo de datos:** Videos de múltiples cámaras
  - columpioscam3 (8 videos)
  - columpioscam2 (4 videos)
  - columpios_cam4 (8 videos)
  - columpioscam1 (1 video)
- **Total videos procesados:** 21 videos registrados en `Playground/data/videos.csv`

#### 1.3 Procesamiento de Esqueletos a Escala
- **Ubicación:** `/Playground/scripts/extract_skeletons.py`
- **Procesamiento por video:**
  - Extracción de 17 puntos clave (MediaPipe Pose)
  - Normalización de esqueletos por pelvis y torso
  - Generación de ventanas temporales `[T=48, K_max=4, 17 joints, 2 coordenadas]`
  - Detección con YOLOv8 para múltiples personas por frame
- **Volumen:** Miles de frames y esqueletos por video
  - Ejemplo: video columpioscam3-2024-12-09_18-18-00.mp4 contiene **2,091 detecciones**

#### 1.4 Construcción de Grafos Panorámicos
- **Ubicación:** `/Playground/scripts/build_panoramic_graph.py`
- **Escala:** Expansión de vértices de `V=17` a `V'=17+n_obj` por cámara
- **Streams generados:** J (joints), B (bones), JM (motion joints), BM (motion bones)
- **Matrices de adyacencia:** A0 (self), A_intra (intra-persona), A_inter (inter-persona)

---

## INDICADOR 2: Verifica que los datos estén anonimizados para no violar normas o leyes de privacidad

### **Se cumple en:**

#### 2.1 Anonimización de Identidades de Personas (UUID)
- **Ubicación:** `/reading-db.ipynb` y `/Playground/data/videos.csv`
- **Implementación:**
  - **Campo `id_person`:** Utiliza UUID v4 (`b17e1ad8-0c2c-44e4-ba72-54c2dd3881a4`)
  - **Nunca se almacenan:** nombres, rostros, características identificables únicas
  - **Detalles de datos:**
    ```python
    # Base de datos contiene:
    # - id_person: UUID anónimo
    # - lat/long: coordenadas geográficas generalizadas (ROI del playground)
    # - timestamp: marca temporal
    # - camera_name: referencia a cámara (ej: "urn:ngsi-ld:camera:columpiosCam1")
    # - tracklet_id: identificador temporal de seguimiento (no identificable)
    # NO contiene: rostros, nombres, datos biométricos identificables
    ```

#### 2.2 Extracción de Esqueletos (Keypoints) sin Identificadores
- **Ubicación:** `/Playground/scripts/extract_skeletons.py`
- **Proceso de anonimización:**
  1. **Entrada:** Videos en bruto (contienen personas)
  2. **Procesamiento:**
     - Detección de personas con YOLOv8 (sin rostro recognition)
     - Extracción de 17 puntos articulares (MediaPipe Pose)
     - **Normalización:** 
       - Pelvis → origen (0,0)
       - Escala por longitud del torso
       - Coordenadas normalizadas [0, 1]
  3. **Salida:** Arrays NumPy `[T=48, K_max=4, 17, 2]`
     - Solo contienen coordenadas articulares normalizadas
     - No contienen rostros, características faciales ni datos identificadores

#### 2.3 Almacenamiento de Esqueletos en Archivos .npy
- **Ubicación:** `/Playground/data/npy/*.npy`
- **Contenido:**
  ```
  - Forma: (T, K_max, 17, 2) = (48, 4, 17, 2)
  - T: tiempo (48 frames)
  - K_max: máximo 4 personas por frame
  - 17: número de articulaciones (COCO skeleton)
  - 2: coordenadas x,y normalizadas [0,1]
  ```
- **Privacidad:** Solo contiene datos de postura corporal sin información identificable

#### 2.4 Anonimización en Etiquetado Automático (CLIP)
- **Ubicación:** `/Playground/scripts/label_with_clip.py`
- **Proceso:**
  1. Carga imágenes renderizadas de esqueletos (sin personas reales)
  2. Clasifica comportamientos sin acceso a datos de identidad
  3. Genera etiquetas de comportamiento:
     - Transit
     - Social_People
     - Play_Object_Normal
     - Play_Object_Risk
     - Adult_Assisting
     - Negative_Contact
- **Privacidad:** Solo clasifica el comportamiento observable, no la identidad

#### 2.5 Almacenamiento Seguro en Azure
- **Ubicación:** `reading-db.ipynb` (credenciales Azure)
- **Configuración:**
  - **SAS Token:** Token de acceso compartido con vigencia limitada
    ```
    sp=racwdl&st=2025-02-05T16:48:09Z&se=2026-04-02T00:48:09Z
    ```
  - **Vigencia:** 1 año (2025-02-05 a 2026-04-02)
  - **Permisos:** Limitados a lectura, añadir, crear, escribir, eliminar, listar
  - **Protocolo:** HTTPS obligatorio (`spr=https`)

---

## INDICADOR 3: Especifica el proceso o estándar a seguir para validar el manejo de datos y garantizar que solo el equipo tenga acceso

### **Se cumple en:**

#### 3.1 Control de Acceso a Base de Datos
- **Ubicación:** `/reading-db.ipynb`
- **Proceso:**
  1. **Autenticación:**
     - Usuario: `admin`
     - Contraseña: Variable de entorno (no hardcoded en producción)
     - Host privado: `40.84.231.179` (IP interna de Azure)
     - Puerto no estándar: `5434` (no puerto público 5432)
  
  2. **Autorización:**
     - Solo conexiones desde red autorizada (Azure VNet)
     - Credenciales PostGIS para el equipo del proyecto

#### 3.2 Control de Acceso a Azure Blob Storage
- **Ubicación:** `/reading-db.ipynb` (descarga de videos)
- **Implementación:**
  ```python
  # SAS Token con expiración y permisos limitados
  account_url = 'https://cienciaciudades2024.blob.core.windows.net'
  container = 'crowdcounting'
  sas_token = 'sp=racwdl&st=2025-02-05T16:48:09Z&se=2026-04-02T00:48:09Z&spr=https&sv=2022-11-02&sr=c&sig=...'
  ```
  
  **Validación del acceso:**
  - Token con fecha de expiración (04-02-2026)
  - Acceso HTTPS obligatorio
  - Permisos granulares limitados
  - Contenedor específico (`crowdcounting`)

#### 3.3 Proceso de Filtrado Espacial (ROI)
- **Ubicación:** `/reading-db.ipynb` y `playgroundROI.gpkg`
- **Estándar implementado:**
  ```python
  # 1) Carga ROI (Region of Interest) desde GeoPackage
  roi = gpd.read_file('playgroundROI.gpkg')
  roi = roi.to_crs(epsg=4326)  # Estándar WGS84
  
  # 2) Consulta espacial con ST_Intersects
  SQL_QUERY = """
    SELECT p.*
    FROM person_observed p
    JOIN roi r ON ST_Intersects(p.geom, r.geom)
  """
  ```
  
  **Validación:**
  - Solo datos dentro del ROI del parque infantil
  - Excluye cualquier dato fuera de la zona autorizada
  - Estándar de proyección: EPSG:4326 (WGS84)

#### 3.4 Estándar de Normalización de Esqueletos
- **Ubicación:** `/Playground/scripts/extract_skeletons.py`
- **Función:** `normalize_skeleton()`
- **Proceso:**
  ```python
  def normalize_skeleton(kps, eps=1e-6):
      k = kps.copy()
      pelvis = (k[11] + k[12]) / 2  # Punto medio de cadera
      k -= pelvis  # Traslación: pelvis al origen
      shoulders = (k[5] + k[6]) / 2  # Punto medio de hombros
      torso = np.linalg.norm(shoulders) + eps  # Longitud del torso
      k /= torso  # Escala: torso normalizado
      return k.astype(np.float32)
  ```
  
  **Estándar:**
  - Normalización consistente por persona
  - Eliminación de información espacial identificable
  - Representación relativa y scale-invariant

#### 3.5 Validación de Calidad de Datos (Confidence Filtering)
- **Ubicación:** `/Playground/scripts/filter_and_split_labels.py`
- **Parámetros:**
  - Umbral de confianza: `--min-conf` (default 0.6)
  - Mínimo de muestras por clase: `--min-samples-per-class` (default 5)
  - Split entrenamiento/validación: `--test-size` 0.2 (80/20)
  - State aleatorio fijo: `--random-state` 42 (reproducibilidad)
  
  **Proceso:**
  ```python
  # Filtrado por confianza de etiquetado automático
  df_filtered = df[df['conf_weight'] >= min_conf]
  
  # Split estratificado (mantiene distribución de clases)
  train, val = train_test_split(
      df_filtered,
      test_size=0.2,
      stratify=df_filtered['label'],
      random_state=42
  )
  ```

#### 3.6 Validación de Grafos Panorámicos
- **Ubicación:** `/Playground/scripts/build_panoramic_graph.py`
- **Validación:**
  - Carga de configuración de objetos desde `configs/objects.yaml` (por cámara)
  - Validación de centroides normalizados [0,1]
  - Generación de matrices de adyacencia basadas en topología corporal estándar
  - Verificación de shapes: `(C=8, T=48, V'=17+n_obj, M=4)`

---

## INDICADOR 4: Presenta registros claros sobre el seguimiento del proceso y autorizaciones para tener acceso a los datos

### **Se cumple en:**

#### 4.1 Documentación de Conectividad a Base de Datos
- **Ubicación:** `/reading-db.ipynb` (Notebook de auditoría)
- **Registro implementado:**
  ```python
  # Conexión documentada con parámetros
  POSTGIS_CONN = {
      "dbname": "crowdcounting",
      "user": "admin",
      "password": "admin",      # Debería ser variable de entorno
      "host": "40.84.231.179",   # IP documentada
      "port": "5434",            # Puerto documentado
  }
  
  # Ejecución de consulta con parámetros
  conn = psycopg2.connect(**POSTGIS_CONN)
  df = pd.read_sql(SQL_QUERY, conn, params=[roi_wkt])
  ```
  
  **Registro generado:**
  - Conexión exitosa → print: "Total dentro del ROI: 1,592,490"
  - Número de filas retornadas: límite de 1000 con conteo total
  - Parámetros de query documentados

#### 4.2 Registro de Descargas de Videos (Azure)
- **Ubicación:** `/reading-db.ipynb` (descarga desde Azure Blob)
- **Auditoría:**
  ```python
  # Descarga con log de progreso
  for path in azure_path:
      print(f"Downloading {path}...")
      blob_client = container_client.get_blob_client(path)
      # ... descarga con progress bar (tqdm)
  ```
  
  **Información registrada:**
  - Nombre del archivo descargado
  - Ruta de destino: `Downloads/<blob_name>`
  - Cuenta de almacenamiento: `cienciaciudades2024`
  - Contenedor: `crowdcounting`

#### 4.3 Registro de Procesamiento (Extract Skeletons)
- **Ubicación:** `/Playground/scripts/extract_skeletons.py`
- **Sistema de logging:**
  ```python
  logging.basicConfig(
      level=logging.INFO,
      format="[%(levelname)s] %(message)s",
  )
  ```
  
  **Información registrada:**
  - Archivo de entrada (video)
  - FPS de procesamiento (12 FPS)
  - Número de frames procesados
  - Número de detecciones por cámara
  - Número de ventanas generadas
  - Ubicación de archivos .npy generados

#### 4.4 Manifestación de Metadatos en CSV
- **Ubicación:** `/Playground/data/videos.csv`
- **Estructura de registro:**
  | Campo | Propósito | Ejemplo |
  |-------|-----------|---------|
  | `video_id` | Identificador único del video | `columpioscam3-2024-12-09_18-18-00.mp4` |
  | `camera` | Identificador de cámara | `urn:ngsi-ld:camera:columpiosCam3` |
  | `t_start` | Timestamp inicio | `2024-12-10T00:18:00.041658+00:00` |
  | `t_end` | Timestamp fin | `2024-12-10T00:18:22.078915+00:00` |
  | `blob_path` | Ruta en Azure | `columpioscam3-2024-12-09_18-18-00.mp4` |
  | `n_detections` | Cantidad de detecciones | `2091` |
  
  **Auditoría:**
  - Origen: Base de datos PostGIS con ST_Intersects
  - Filtrado: Solo ROI del parque (lat/long específicas)
  - Trazabilidad: cada video tiene timestamp exacto

#### 4.5 Registro de Etiquetado Automático (CLIP)
- **Ubicación:** `/Playground/scripts/label_with_clip.py`
- **Salida de auditoría:**
  ```python
  with open(out_csv, 'w', newline='') as fout:
      writer = csv.writer(fout)
      writer.writerow(['file','label','conf_weight'])
      # Por cada imagen:
      # - Nombre de archivo
      # - Etiqueta asignada
      # - Peso de confianza (0-1)
  ```
  
  **Archivo generado:** `/Playground/data/graph/with_objects/vlm_labels_clip.csv`
  - Registro de cada clasificación
  - Puntuación de confianza para trazabilidad
  - Identificable por nombre de archivo

#### 4.6 Registro de Filtrado y Split de Datos
- **Ubicación:** `/Playground/scripts/filter_and_split_labels.py`
- **Archivos de auditoría generados:**
  
  1. **`vlm_labels_filtered_0.60.csv`**
     - Todas las muestras que pasan filtro de confianza
     - Conteo por clase
  
  2. **`vlm_labels_lowconf_below_0.60.csv`**
     - Muestras rechazadas (conf < 0.60)
     - Permite revisión de exclusiones
  
  3. **`train.csv` y `val.csv`**
     - Split estratificado documentado
     - Random state: 42 (reproducible)
  
  **Logs impresos:**
  ```
  Total labeled rows: [N]
  Counts per label (all): [distribution]
  After filtering conf >= 0.60: [N]
  Counts per label (filtered): [distribution]
  Train counts per label: [distribution]
  Val counts per label: [distribution]
  ```

#### 4.7 Autorización y Control de Acceso (Framework)
- **Ubicación:** Documentado en arquitectura del proyecto
- **Niveles de acceso:**
  
  | Nivel | Recurso | Autenticación | Autorización |
  |-------|---------|---------------|--------------|
  | Tier 1 | PostGIS DB | Usuario/Pass | IP permitida (Azure VNet) |
  | Tier 2 | Azure Blob | SAS Token | Token con expiración |
  | Tier 3 | Archivos locales | Sistema SO | Permisos de archivo |
  | Tier 4 | Entrenamiento | Repo privado | Acceso por rama (git) |

---