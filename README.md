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
Official Implementation of Barbershop. **KEEP UPDATING! Please Git Pull the latest version.**

## Updates
`2021/12/27` Add dilation and erosion parameters to smooth the boundary.

#### `2021/12/24` Important Update: Add semantic mask inpainting module to solve the occlusion problem. Please git pull the latest version.

`2021/12/18` Add a rough version of the project.

`2021/06/02` Add project page.


## Installation
- Clone the repository:
``` 
git clone https://github.com/ZPdesu/Barbershop.git
cd Barbershop
```
- Dependencies:  
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). 
All dependencies for defining the environment are provided in `environment/environment.yaml`.


## Download II2S images
Please download the [II2S](https://drive.google.com/drive/folders/15jsR9yy_pfDHiS9aE3HcYDgwtBbAneId?usp=sharing) 
and put them in the `input/face` folder.


## Getting Started  
Preprocess your own images. Please put the raw images in the `unprocessed` folder.
```
python align_face.py
```

Produce realistic results:
```
python main.py --im_path1 90.png --im_path2 15.png --im_path3 117.png --sign realistic --smooth 5
```

Produce results faithful to the masks:
```
python main.py --im_path1 90.png --im_path2 15.png --im_path3 117.png --sign fidelity --smooth 5
```



## Todo List
* add a detailed readme
* update mask inpainting code
* integrate image encoder
* add preprocessing step
* ...

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
