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
Official Implementation of Barbershop.


<span style="color:red">**KEEP UPDATING !**</span>


Option1: Produce realistic results:
```
python main.py --im_path1 90.png --im_path2 15.png --im_path3 117.png --sign realistic
```

Option2: Produce results faithful to the masks:
```
python main.py --im_path1 90.png --im_path2 15.png --im_path3 117.png --sign fidelity
```




## <span style="color:red"> Updates</span>
#### <span style="color:red">24/12/2021 Important Update: Add improved semantic mask inpainting module. Please git pull the newest version.</span>

**18/12/2021** Add a rough version of the project.

**2/6/2021** Add project page.


## Todo List
* add a detailed readme
* update code
* integrate image encoder
* add preprocessing step
* ...


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
