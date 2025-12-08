import os
import numpy as np

DIR = "data/npy"      # carpeta donde están tus npy y npz
PREFIX = "columpioscam3-2024-12-09_18-18-00.mp4_win"  # prefijo del video que estamos analizando

print(f"Analizando NPZ (meta) desde win000 hasta win019...\n")

for i in range(20):
    win = f"{i:03d}"
    fname = os.path.join(DIR, f"{PREFIX}{win}_meta.npz")
    
    if not os.path.exists(fname):
        print(f"[{win}] No existe: {fname}")
        continue

    print(f"\n=== {os.path.basename(fname)} ===")

    try:
        data = np.load(fname, allow_pickle=True)
    except Exception as e:
        print(" Error al cargar:", e)
        continue

    keys = list(data.keys())
    print("Claves dentro del NPZ:", keys)

    # Ahora revisamos cada array
    for k in keys:
        arr = data[k]
        print(f"--- {k} ---")
        
        if isinstance(arr, np.ndarray):
            print(" shape:", arr.shape)
            try:
                print(" min:", arr.min(), " max:", arr.max())
            except:
                print(" (no se puede obtener min/max)")
        else:
            print(" (no es array numpy; tipo:", type(arr), ")")

    print("\n")
