# Matriz de Trazabilidad: Privacidad y Control de Acceso a Datos
## Claves de Acceso Documentadas

### Base de Datos PostGIS
```
Host: 40.84.231.179 (Azure Private IP)
Puerto: 5434 (no estándar)
Base Datos: crowdcounting
Usuario: admin
Ubicación: /reading-db.ipynb
Acceso: Solo desde Azure VNet
```

### Azure Blob Storage
```
Cuenta: cienciaciudades2024.blob.core.windows.net
Contenedor: crowdcounting
Autenticación: SAS Token
Vigencia: 2025-02-05 to 2026-04-02
Protocolo: HTTPS (forzado)
Ubicación: /reading-db.ipynb
```

### Archivos Locales
```
Datos: /Playground/data/
  - videos.csv (manifest de videos)
  - npy/ (esqueletos procesados)
  - graph/ (grafos y etiquetas)
Acceso: Permisos del sistema operativo
Documentación: README.md en cada carpeta
```

---

## La seguridad y privacidad de los datos
Los datos no son estrictamente privados debido a que los recursos dados por el socio formador, accedidos mediante credenciales que estan publicas dentro del archivo publico 'reading-db.ipynb' del socioformador pueden ser extraidos y usados por terceros. 

Debido a esta vulnerabilidad de los datos. Los scripts CLI generados por el equipo de desarrollo, como es el caso de download_videos.py, requieren que las credenciales sean pasadas en la ejecuccion CLI mediante una variable de ambiente o escribir directamente las credenciales en linea de comando.

El script de etiquetado con uso de VLM tampoco utiliza APIs y no se sube informacion a distintos lados, se utilizo un modelo preentrenado descargado con capacidad de reconocimiento de lenguaje natural para el etiquetado de datos, manteniendo asi la informacion de forma local para evitar fugas de informacion.

## Conclusion
La seguridad y privacidad de los datos han sido cuidados por el equipo de desarrollo en los archivos nuevos creados por el equipo de desarrollo.