# Solución del reto — Réplica del Paper MP‑GCN

Este repositorio recoge la primera parte del reto planteado en el curso **Socio‑Formador IA Avanzada**, en la que se pide replicar el paper *“Skeleton‑based Group Activity Recognition via Spatial‑Temporal Panoramic Graph”*.  El objetivo es comprender cómo funciona el modelo MP‑GCN y preparar un entorno de trabajo antes de procesar nuestros propios datos de parque infantil【321910613868558†L257-L304】.

## ¿Qué contiene este repositorio?

* **`hello_mpgcn.ipynb`** – Un cuaderno Jupyter que instala las dependencias necesarias, construye un conjunto de datos sintético con la forma correcta \([C,T,V',M]\), define las matrices de adyacencia para las conexiones intra‑persona, persona↔objeto e inter‑persona, implementa una versión mínima de MP‑GCN con cuatro streams (Joints, Bones, Joint Motion y Bone Motion) y ejecuta un pase hacia adelante con datos sintéticos.  Este cuaderno sirve para familiarizarse con la estructura de entrada y el flujo de datos del modelo original.
* **`README.md`** – El archivo que estás leyendo, que resume el propósito del repositorio y explica brevemente el contenido del cuaderno.

## Resumen de la implementación

El cuaderno `hello_mpgcn.ipynb` sigue estos pasos:

1. **Instalación de dependencias** – Utiliza `pip` para instalar `torch`, `pyyaml`, `tqdm`, `tensorboardX` y `fvcore`, tal y como recomienda el repositorio oficial de MP‑GCN【247470307142095†L261-L276】.
2. **Construcción de datos sintéticos** – Se genera un tensor aleatorio con forma \([C,T,V',M]\) donde `C=3` (canales: coordenada *x*, coordenada *y*, confianza), `T=48` (marcos temporales), `V'=18` (17 articulaciones humanas + 1 objeto) y `M=4` (máximo de personas).  Esto simula la estructura de entrada que el feeder del paper prepara a partir de poses humanas y objetos.
3. **Definición de las matrices de adyacencia** – Se crean las matrices `A0` (identidad), `A_intra` (conexiones del esqueleto humano y enlaces manos↔objeto) y `A_inter` (conexiones de pelvis entre personas).  Estas matrices codifican la topología del grafo panorámico descrito en el artículo【321910613868558†L257-L304】.
4. **Modelo simplificado** – Se define una clase `SimpleMPGCN` que aplica convoluciones espaciales a cada stream (J, B, JM, BM) y después fusiona los resultados para obtener logits.  Aunque no es una implementación completa, reproduce la idea central de usar cuatro flujos separados y un grafo panorámico para capturar interacciones humano‑humano y humano‑objeto.
5. **Ejecución y resultados** – Se realiza un pase hacia adelante con los datos sintéticos y se imprime la forma de la salida.  De esta manera se verifica que el pipeline básico funciona correctamente y que las formas de los tensores son las esperadas.

## Próximos pasos

Para replicar íntegramente el paper, sería necesario descargar los conjuntos de datos oficiales (Volleyball, NBA o Kinetics) y preparar los datos de acuerdo con el README del repositorio original【247470307142095†L276-L336】.  Este repositorio sienta las bases para esa fase final, familiarizando al estudiante con el modelo y la estructura de datos sin requerir descargas pesadas.
