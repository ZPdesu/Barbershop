# Barbershop: GAN-based Image Compositing using Segmentation Masks
![teaser](docs/assets/teaser.png)

> [**Barbershop: GAN-based Image Compositing using Segmentation Masks**](https://zpdesu.github.io/Barbershop/)<br/>
[Peihao Zhu](https://github.com/ZPdesu),
[Rameen Abdal](https://github.com/RameenAbdal),
[John Femiani](https://scholar.google.com/citations?user=rS1xJIIAAAAJ&hl=en),
[Peter Wonka](http://peterwonka.net/)<br/>


> [arXiv](https://arxiv.org/abs/2106.01505) | [BibTeX](#bibtex) | [Project Page](https://zpdesu.github.io/Barbershop/) | [Video](https://youtu.be/ZU-yrAvoJfQ)


> **Abstract** Seamlessly blending features from multiple images is extremely challenging because of complex relationships in lighting, geometry, and partial occlusion which cause coupling between different parts of the image. Even though recent work on GANs enables synthesis of realistic hair or faces, it remains difficult to combine them into a single, coherent, and plausible image rather than a disjointed set of image patches. We present a novel solution to image blending, particularly for the problem of hairstyle transfer, based on GAN-inversion. We propose a novel latent space for image blending which is better at preserving detail and encoding spatial information, and propose a new GAN-embedding algorithm which is able to slightly modify images to conform to a common segmentation mask. Our novel representation enables the transfer of the visual properties from multiple reference images including specific details such as moles and wrinkles, and because we do image blending in a latent-space  we are able to synthesize images that are coherent. Our approach avoids blending artifacts present in other approaches and finds a globally consistent image. Our results demonstrate a significant improvement over the current state of the art in a user study, with users preferring our blending solution over 95 percent of the time.


## Description

This repository is a fork of the [official implmentation of Barbershop](https://github.com/ZPdesu/Barbershop). This repository build on the official reporsitory to add the following features:

- Combine [`main.py`](https://github.com/ZPdesu/Barbershop/blob/main/main.py) and [`align_face.py`](https://github.com/ZPdesu/Barbershop/blob/main/align_face.py) into a single command line interface as part of the updated [`main.py`](https://github.com/soumik12345/Barbershop/blob/main/main.py).
- Provide a notebook [`inference.ipynb`](https://github.com/soumik12345/Barbershop/blob/main/inference.ipynb) for performing step-by-step inference and visualization of the result.
- Add an integration with Weights & Biases, which enables the predictions to be visualized as a W&B Table. The integration works with both the script and the notebook.

## Installation
- Clone the repository:
``` 
git clone https://github.com/ZPdesu/Barbershop.git
cd Barbershop
```
- Dependencies:  
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). 
All dependencies for defining the environment are provided in `environment/environment.yaml`.

## Getting Started
Produce realistic results:
```
python main.py --identity_image 90.png --structure_image 15.png --appearance_image 117.png --sign realistic --smooth 5
```

Produce results faithful to the masks:
```
python main.py --identity_image 90.png --structure_image 15.png --appearance_image 117.png --sign fidelity --smooth 5
```

You can also use the [Jupyter Notebook](./inference.ipynb) to producde the results. The results are now logged automatically as a Weights and Biases Table.

![](https://i.imgur.com/subthu8.png)

## Acknowledgments
This code borrows heavily from [II2S](https://github.com/ZPdesu/II2S).

## BibTeX

```
@misc{zhu2021barbershop,
      title={Barbershop: GAN-based Image Compositing using Segmentation Masks},
      author={Peihao Zhu and Rameen Abdal and John Femiani and Peter Wonka},
      year={2021},
      eprint={2106.01505},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
