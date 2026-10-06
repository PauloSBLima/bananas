import os
from ultralytics import YOLO

BASE_DIR = r"C:\Users\Deborah\Documents\VSCode\MLVC\bananas"

pastas_imagens = [
    os.path.join(BASE_DIR, "images", "train", "maduras"),
    os.path.join(BASE_DIR, "images", "val", "maduras")
]

print("Carregando YOLO-World...")
model = YOLO("yolov8s-world.pt")
model.set_classes(["banana"])

confianca = 0.25

print("\nIniciando o Auto-Labeling...\n")

for pasta_img in pastas_imagens:
    print(f"Verificando pasta: {pasta_img}")
    if not os.path.exists(pasta_img):
        print("  -> ERRO: A pasta não existe no disco!")
        continue
        
    arquivos = os.listdir(pasta_img)
    print(f"  -> Arquivos encontrados na pasta: {arquivos}")
        
    pasta_label = pasta_img.replace("images", "labels")
    os.makedirs(pasta_label, exist_ok=True)
    
    for nome_arquivo in arquivos:
        # Adicionado suporte a extensões maiúsculas (.JPG, .PNG)
        if nome_arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            caminho_imagem = os.path.join(pasta_img, nome_arquivo)
            
            nome_txt = os.path.splitext(nome_arquivo)[0] + ".txt"
            caminho_txt = os.path.join(pasta_label, nome_txt)
            
            if os.path.exists(caminho_txt):
                print(f"  -> Já existe label para: {nome_arquivo}")
                continue
            
            print(f"Processando imagem: {nome_arquivo}...")
            resultados = model.predict(caminho_imagem, conf=confianca, verbose=False)
            
            with open(caminho_txt, 'w') as f:
                for box in resultados[0].boxes:
                    cls_id = int(box.cls[0])          
                    x_c, y_c, w, h = box.xywhn[0]      
                    f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            
            print(f"  -> Marcado com sucesso! Encontrou {len(resultados[0].boxes)} objeto(s).")

print("\nFim da verificação.")