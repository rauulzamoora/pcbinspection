from ultralytics import YOLO
import cv2
import os
import glob
import tkinter as tk
from tkinter import filedialog

def inspeccion_manual():
    # --- 1. CARGAR MODELO ---
    ruta_modelo = r"C:\Users\raulz\runs\detect\pcb_exito6\weights\best.pt"
    
    if not os.path.exists(ruta_modelo):
        print(f"❌ Error: No encuentro el modelo en {ruta_modelo}")
        return

    print("Cargando modelo...")
    model = YOLO(ruta_modelo)

    # --- 2. SELECCIONAR CARPETA ---
    root = tk.Tk()
    root.withdraw()
    print("Selecciona la carpeta con las imágenes...")
    carpeta = filedialog.askdirectory(title="Selecciona carpeta de imágenes PCB")
    root.destroy()

    if not carpeta:
        print("Operación cancelada.")
        return

    # Buscar imágenes
    exts = ('*.jpg', '*.jpeg', '*.png')
    lista_imagenes = []
    for ext in exts:
        lista_imagenes.extend(glob.glob(os.path.join(carpeta, "**", ext), recursive=True))

    if not lista_imagenes:
        print("❌ No se encontraron imágenes.")
        return

    print(f"Cargadas {len(lista_imagenes)} imágenes.")
    print("\n---------------- CONTROLES ----------------")
    print(" [D]  -> Siguiente Imagen")
    print(" [A]  -> Imagen Anterior")
    print(" [Q]  -> Salir del programa")
    print("-------------------------------------------")

    # --- 3. BUCLE DE NAVEGACIÓN ---
    index = 0
    total = len(lista_imagenes)

    while True:
        # Asegurar que el índice sea válido (navegación circular)
        if index >= total: index = 0
        if index < 0: index = total - 1

        ruta_img = lista_imagenes[index]
        nombre_archivo = os.path.basename(ruta_img)

        # --- PREDICCIÓN ---
        # verbose=False evita que llene la consola de texto técnico
        results = model.predict(source=ruta_img, conf=0.35, iou=0.5, save=False, verbose=False)
        
        # --- GENERAR IMAGEN VISUAL ---
        annotated_frame = results[0].plot(line_width=2, font_size=1)

        # --- INFORMACIÓN EN PANTALLA ---
        # Panel superior negro para que se lea el texto
        cv2.rectangle(annotated_frame, (0, 0), (1000, 60), (0,0,0), -1)
        
        texto_nav = f"Imagen {index+1}/{total}: {nombre_archivo}"
        texto_ayuda = "[A] <- Anterior | [D] -> Siguiente | [Q] Salir"
        
        cv2.putText(annotated_frame, texto_nav, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, texto_ayuda, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Mostrar imagen
        cv2.imshow("Inspector PCB Manual", annotated_frame)

        # --- IMPRIMIR REPORTE EN CONSOLA (Solo una vez por imagen) ---
        # Limpiamos consola o separamos para que sea legible
        print(f"\n[{index+1}/{total}] Reporte: {nombre_archivo}")
        hallazgos = 0
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            nombre_defecto = model.names[cls_id]
            print(f"   ⚠️  {nombre_defecto.upper()} ({conf:.1%})")
            hallazgos += 1
        if hallazgos == 0: print("   ✅ Sin defectos.")

        # --- ESPERAR TECLA (Control Manual) ---
        # waitKey(0) congela el programa hasta que toques una tecla
        key = cv2.waitKey(0)

        # Lógica de teclas
        if key == ord('d') or key == ord('D') or key == 83: # 83 es flecha derecha en algunos sistemas
            index += 1
            print(">>> Siguiente")
            
        elif key == ord('a') or key == ord('A') or key == 81: # 81 es flecha izquierda
            index -= 1
            print("<<< Anterior")
            
        elif key == ord('q') or key == ord('Q') or key == 27: # 27 es ESC
            print("\n🛑 Programa finalizado.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    inspeccion_manual()