import os
import glob
import shutil
import cv2
import random
from tqdm import tqdm

# --- RUTAS ---
BASE_DIR = r"C:\Users\raulz\Desktop\PCB inspection"
RAW_DATA_PATH = os.path.join(BASE_DIR, "DeepPCB_Raw")
DEST_PATH = os.path.join(BASE_DIR, "datasets")

# Mapeo según la documentación oficial de DeepPCB
# 1-open, 2-short, 3-mousebite, 4-spur, 5-copper, 6-pin-hole
# YOLO necesita empezar en 0, así que restaremos 1 al ID.
CLASSES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']

def convert_box(size, box):
    # Convertir x1,y1,x2,y2 a Center_x, Center_y, Width, Height normalizado
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (x * dw, y * dh, w * dw, h * dh)

def main():
    print(f"INICIANDO PROCESAMIENTO")
    
    # 1. Buscar imágenes
    raw_images = glob.glob(os.path.join(RAW_DATA_PATH, "**", "*_test.jpg"), recursive=True)
    if not raw_images:
        print("❌ ERROR: No encontré imágenes en DeepPCB_Raw.")
        return

    print(f"Se encontraron {len(raw_images)} imágenes candidatas.")

    # 2. Limpiar destino
    if os.path.exists(DEST_PATH):
        try:
            shutil.rmtree(DEST_PATH)
        except:
            print("⚠️ Advertencia: No se pudo borrar carpeta anterior.")

    # 3. Crear carpetas
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DEST_PATH, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(DEST_PATH, 'labels', split), exist_ok=True)

    # 4. Procesar
    random.seed(42)
    random.shuffle(raw_images)
    split_idx = int(len(raw_images) * 0.8)
    splits = {'train': raw_images[:split_idx], 'val': raw_images[split_idx:]}
    
    conteo_exitos = 0
    conteo_sin_txt = 0

    for subset, img_list in splits.items():
        for img_path in tqdm(img_list, desc=f"Procesando {subset}"):
            
            # --- 1. ENCONTRAR LA PAREJA TXT ---
            carpeta = os.path.dirname(img_path)
            nombre_img = os.path.basename(img_path) # ej: 00041000_test.jpg
            
            # El ID es lo que está antes del primer guion bajo
            file_id = nombre_img.split('_')[0] # ej: 00041000
            
            # La documentación dice que el archivo se llama "00041000.txt"
            src_txt = os.path.join(carpeta, f"{file_id}.txt")

            if not os.path.exists(src_txt):
                conteo_sin_txt += 1
                continue

            # --- 2. LEER Y CONVERTIR SEGÚN FORMATO OFICIAL ---
            img = cv2.imread(img_path)
            if img is None: continue
            h, w = img.shape[:2]

            labels_yolo = []
            with open(src_txt, 'r') as f:
                for line in f:
                    # La documentación dice: x1,y1,x2,y2,type
                    # A veces usan espacios, a veces comas. Reemplazamos comas por espacios para estar seguros.
                    parts = line.replace(',', ' ').strip().split()
                    
                    if len(parts) < 5: continue
                    
                    try:
                        x1, y1, x2, y2, type_id = map(int, parts[:5])
                        
                        # FILTRO: type_id 0 es background (no usado)
                        if type_id < 1 or type_id > 6:
                            continue
                        
                        # CONVERSIÓN DE ID: DeepPCB (1-6) -> YOLO (0-5)
                        yolo_id = type_id - 1
                        
                        # CONVERSIÓN DE CAJA
                        bbox = convert_box((w, h), (x1, y1, x2, y2))
                        
                        labels_yolo.append(f"{yolo_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
                    except ValueError:
                        continue # Si hay algo que no es número, saltar línea

            # --- 3. GUARDAR SOLO SI HAY ETIQUETAS ---
            if labels_yolo:
                # Copiar imagen
                dst_img = os.path.join(DEST_PATH, 'images', subset, nombre_img)
                shutil.copy(img_path, dst_img)
                
                # Guardar txt nuevo
                dst_txt = os.path.join(DEST_PATH, 'labels', subset, nombre_img.replace('.jpg', '.txt'))
                with open(dst_txt, 'w') as f:
                    f.write('\n'.join(labels_yolo))
                
                conteo_exitos += 1

    print("\n" + "="*40)
    print(f"REPORTE FINAL")
    print(f"✅ Pares generados correctamente: {conteo_exitos}")
    if conteo_sin_txt > 0:
        print(f"⚠️ Imágenes ignoradas (sin txt): {conteo_sin_txt}")
    print("="*40)
    
    # Crear data.yaml
    path_root = DEST_PATH.replace('\\', '/')
    yaml_content = f"""path: {path_root}
train: images/train
val: images/val
nc: 6
names: {CLASSES}
"""
    with open(os.path.join(DEST_PATH, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    main()