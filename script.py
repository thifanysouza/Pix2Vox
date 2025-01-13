import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from mpl_toolkits.mplot3d import Axes3D
import cv2
import time  # Importação adicional para medir o tempo

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger
from utils.helpers import get_volume_views
from utils.data_transforms import Compose, ToTensor, Normalize, CenterCrop, RandomBackground, ColorJitter, RandomNoise, RandomFlip
from config import cfg

def load_images():
    """Permite ao usuário carregar uma ou mais imagens."""
    Tk().withdraw()
    file_paths = filedialog.askopenfilenames(title="Selecione as imagens", filetypes=[("Imagens", "*.png;*.jpg;*.jpeg")])
    return file_paths

def preprocess_image(image_path):
    """Pré-processa uma imagem para entrada no modelo usando transformações compostas."""
    img_size = (cfg.CONST.IMG_H, cfg.CONST.IMG_W)
    crop_size = (cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W)
    mean = cfg.DATASET.MEAN
    std = cfg.DATASET.STD
    random_bg_range = cfg.TEST.RANDOM_BG_COLOR_RANGE

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0  # Normalizar para [0, 1]

    # Compor transformações
    transform = Compose([
        CenterCrop(img_size, crop_size),
        RandomBackground(random_bg_range),
        ColorJitter(brightness=cfg.TRAIN.BRIGHTNESS, contrast=cfg.TRAIN.CONTRAST, saturation=cfg.TRAIN.SATURATION),
        RandomNoise(noise_std=cfg.TRAIN.NOISE_STD),
        RandomFlip(),
        Normalize(mean=mean, std=std),
        ToTensor()
    ])

    # Aplicar transformações
    image = transform(np.array([image]))  # Adicionar batch dimension
    return image

def save_obj(volume, file_path):
    """Salva o volume 3D binarizado como um arquivo .obj."""
    vertices = []
    faces = []
    index_map = {}

    # Converter volume em vértices e faces
    for x in range(volume.shape[0]):
        for y in range(volume.shape[1]):
            for z in range(volume.shape[2]):
                if volume[x, y, z] > 0:  # Checa explicitamente se o voxel é verdadeiro
                    # Adicionar vértices
                    v_idx = len(vertices) + 1
                    index_map[(x, y, z)] = v_idx
                    vertices.append((x, y, z))

    # Adicionar faces conectando vértices vizinhos
    for (x, y, z), v_idx in index_map.items():
        for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            neighbor = (x + dx, y + dy, z + dz)
            if neighbor in index_map:
                faces.append((v_idx, index_map[neighbor]))

    # Salvar em formato .obj
    with open(file_path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]}\n")

def visualize_3d(volume):
    """Visualiza o objeto reconstruído usando matplotlib."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(volume, facecolors='blue', edgecolor='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def remove_module_prefix(state_dict):
    """Remove o prefixo 'module.' das chaves no state_dict."""
    return {k.replace('module.', ''): v for k, v in state_dict.items()}

def main():
    # Configurações
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_path = "models/pre-trained/Pix2Vox++-A-ShapeNet.pth"

    # Inicializar modelos
    encoder = Encoder(cfg).to(device).eval()
    decoder = Decoder(cfg).to(device).eval()
    refiner = Refiner(cfg).to(device).eval() if cfg.NETWORK.USE_REFINER else None
    merger = Merger(cfg).to(device).eval() if cfg.NETWORK.USE_MERGER else None

    # Carregar pesos pré-treinados
    checkpoint = torch.load(weights_path, map_location=device)
    checkpoint['encoder_state_dict'] = remove_module_prefix(checkpoint['encoder_state_dict'])
    checkpoint['decoder_state_dict'] = remove_module_prefix(checkpoint['decoder_state_dict'])
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    if refiner and 'refiner_state_dict' in checkpoint:
        checkpoint['refiner_state_dict'] = remove_module_prefix(checkpoint['refiner_state_dict'])
        refiner.load_state_dict(checkpoint['refiner_state_dict'])

    if merger and 'merger_state_dict' in checkpoint:
        checkpoint['merger_state_dict'] = remove_module_prefix(checkpoint['merger_state_dict'])
        merger.load_state_dict(checkpoint['merger_state_dict'])

    # Carregar imagens
    image_paths = load_images()
    start_time = time.time()
    if not image_paths:
        print("Nenhuma imagem selecionada. Encerrando.")
        return

    for image_path in image_paths:
        print(f"Processando imagem: {image_path}")

        # Pré-processar imagem
        image = preprocess_image(image_path).to(device)

        # Adicionar dimensões para compatibilidade com o modelo
        image = image.unsqueeze(0).permute(1, 0, 2, 3, 4)  # (n_views, batch_size, channels, height, width)

        # Inferência
        with torch.no_grad():
            image_features = encoder(image)
            raw_features, coarse_volume = decoder(image_features)

            if merger:
                coarse_volume = merger(raw_features, coarse_volume)

            if refiner:
                refined_volume = refiner(coarse_volume)
            else:
                refined_volume = coarse_volume

            volume = refined_volume.squeeze(0).cpu().numpy() > 0.5  # Binarizar o volume

        # Salvar como .obj
        output_path = os.path.splitext(image_path)[0] + ".obj"
        save_obj(volume, output_path)
        print(f"Objeto 3D salvo em: {output_path}")

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Tempo total de execução: {total_time:.2f} segundos")

        # Visualizar
        visualize_3d(volume)

if __name__ == "__main__":
    main()










