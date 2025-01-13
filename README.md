# Avaliação do modelo Pix2Vox++

O presente repositório foi desenvolvido para avaliar a capacidade da candidata em trabalhar com modelos de
machine learning para recriação de objetos 3D a partir de imagens 2D, utilizando ferramentas e modelos pré-treinados. 

O trabalho desenvolvido teve como base o artigo [Pix2Vox++: Multi-scale Context-aware 3D Object Reconstruction from Single and Multiple Images](https://arxiv.org/abs/2006.12250). 

![Overview](https://www.infinitescript.com/projects/Pix2Vox/Pix2Vox++-Overview.jpg)

## Referência

```
@article{xie2020pix2vox,
  title={Pix2Vox++: Multi-scale Context-aware 3D Object Reconstruction from Single and Multiple Images},
  author={Xie, Haozhe and 
          Yao, Hongxun and 
          Zhang, Shengping and 
          Zhou, Shangchen and 
          Sun, Wenxiu},
  journal={International Journal of Computer Vision (IJCV)},
  year={2020},
  doi={10.1007/s11263-020-01347-6}
}
```

## Dataset para etapa de teste

- Pix3D images & voxelized models: http://pix3d.csail.mit.edu/data/pix3d.zip

## Modelo pré-treinado

Modelo pré-treinado em ShapeNet disponível em:

- [Pix2Vox++/A](https://gateway.infinitescript.com/?fileName=Pix2Vox%2B%2B-A-ShapeNet.pth) (385.4 MB)

Para correta execução do script, adicione o arquivo .pth no seguinte caminho:
```
./models/pre-trained/Pix2Vox++-A-ShapeNet.pth
```


## Pre - requisitos

#### Clone o repositório

```
git clone https://github.com/thifanysouza/Pix2Vox.git
```

#### Instalação de dependências

```
cd Pix2Vox
pip install -r requirements.txt
```


## Get Started

Execução do teste do modelo:

```
python runner.py --test --weights=path/to/model.pth
```

Execução do script desenvolvido:

```
python script.py
```