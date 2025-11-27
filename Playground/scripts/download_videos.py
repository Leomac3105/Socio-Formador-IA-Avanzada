#!/usr/bin/env python3
"""Descarga los videos listados en data/videos.csv desde Azure Blob Storage."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def get_blob_client(
    account_name: str,
    container: str,
    connection_string: str | None,
    sas_token: str | None = None,
) -> BlobServiceClient:
    """Obtiene un cliente autenticado.

    Prioridad de autenticación:
      1. `connection_string` (si se pasa)
      2. `sas_token` (si se pasa)
      3. `AZURE_STORAGE_ACCOUNT_KEY` env var
      4. `AZURE_STORAGE_CONNECTION_STRING` env var (implicitamente manejado por from_connection_string)
    """
    if connection_string:
        return BlobServiceClient.from_connection_string(connection_string)

    if sas_token:
        account_url = f"https://{account_name}.blob.core.windows.net"
        return BlobServiceClient(account_url=account_url, credential=sas_token)

    account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
    if account_key:
        account_url = f"https://{account_name}.blob.core.windows.net"
        return BlobServiceClient(account_url=account_url, credential=account_key)

    # fall back: look for AZURE_STORAGE_CONNECTION_STRING env var
    conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if conn:
        return BlobServiceClient.from_connection_string(conn)

    raise ValueError("Define --connection-string, --sas-token o la variable de entorno AZURE_STORAGE_ACCOUNT_KEY / AZURE_STORAGE_CONNECTION_STRING")


def download_blob(
    blob_service: BlobServiceClient,
    container: str,
    blob_path: str,
    local_path: Path,
) -> bool:
    """Descarga un blob a la ruta local."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        blob_client = blob_service.get_blob_client(container=container, blob=blob_path)
        with open(local_path, "wb") as handle:
            handle.write(blob_client.download_blob().readall())
        return True
    except ResourceExistsError:
        logging.info("Ya existe: %s", local_path)
        return True
    except Exception as exc:  # noqa: BLE001
        logging.error("Error descargando %s -> %s", blob_path, exc)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga los videos requeridos para el pipeline Playground.")
    parser.add_argument("--videos-csv", type=Path, default=Path("data/videos.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("Downloads"))
    parser.add_argument("--account-name", type=str, default="socioformadorstorage")
    parser.add_argument("--container", type=str, default="videos")
    parser.add_argument("--connection-string", type=str, default=os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
    parser.add_argument("--sas-token", type=str, default=None, help="SAS token para acceso a blobs (p. ej. 'sp=...&st=...&se=...')")
    args = parser.parse_args()

    if not args.videos_csv.exists():
        raise FileNotFoundError(f"No existe {args.videos_csv}")

    df = pd.read_csv(args.videos_csv)
    blob_client = get_blob_client(args.account_name, args.container, args.connection_string, args.sas_token)

    success = 0
    def sanitize_camera(cam: str) -> str:
        # Extrae la parte legible del identificador de cámara (p. ej. urn:...:columpiosCam2 -> columpiosCam2)
        if isinstance(cam, str) and ":" in cam:
            return cam.split(":")[-1]
        # fallback: reemplaza caracteres no alfanuméricos por guion bajo
        return "".join([c if c.isalnum() else "_" for c in str(cam)])

    for _, row in df.iterrows():
        blob_path = row["blob_path"]
        camera = row.get("camera", "unknown_camera")
        cam_folder = sanitize_camera(camera)
        local_path = args.output_dir / cam_folder / Path(blob_path).name
        if local_path.exists():
            logging.info("Saltando (ya descargado): %s", local_path)
            success += 1
            continue
        if download_blob(blob_client, args.container, blob_path, local_path):
            logging.info("Descargado: %s", local_path)
            success += 1

    logging.info("Videos descargados correctamente: %d/%d", success, len(df))


if __name__ == "__main__":
    main()
