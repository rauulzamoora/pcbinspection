from ultralytics import YOLO
import torch
import os

def main():
    print(f"Usando dispositivo: {torch.cuda.get_device_name(0)}")
    
    # RUTA FIJA Y ABSOLUTA AL YAML
    yaml_path = r"C:\Users\raulz\Desktop\PCB inspection\datasets\data.yaml"
    
    # Comprobación de seguridad visual
    if not os.path.exists(yaml_path):
        print(f"❌ ERROR: No existe el archivo: {yaml_path}")
        return

    model = YOLO('yolov8n.pt') 
    
    model.train(
        data=yaml_path, # Nombre del archivo  
        epochs=50, # Número de barridos
        imgsz=640, # Tamaño de dimensionamiento
        batch=8, # Número de imagenes a la vez         
        device=0, # Motor de NVIDIA
        workers=0, # Número de subprocesos
        name='pcb_exito' # Nombre de exportación
    )

if __name__ == '__main__':
    main()