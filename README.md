# CCAgT-utils

CCAgT-utils it's a package to work with the **CCAgT dataset**: `Images of Cervical Cells with AgNOR Stain Technique`. The package will provide some customized codes for annotations format conversion, mask generation, plotting samples, etc.


## Package context
I have been working with images of cervical cells stained with AgNOR since January/2020 for my master thesis. The results of my thesis you can find at [CCAgT-benchmarks](https://github.com/johnnv1/CCAgT-benchmarks). In general, the objective of the thesis it's automatize the principal part to help at the diagnostic/prognostic of these cells. Therefore, I also have developed some codes to preprocess or just to help in the use of this dataset.


These codes to work with the dataset will be available at this package.

## Contents

1. [Links to download the dataset](#links-to-download-the-ccagt-dataset)
2. [What is this dataset like?](#what-is-this-dataset-looks-like)
3. [Examples of use of this package](#examples-of-use)


# Links to download the CCAgT dataset

1. Version 1.1 - [drive](https://drive.google.com/drive/folders/1TBpYCv6S1ydASLauSzcsvO7Wc5O-WUw0?usp=sharing) or [UFSC repository](https://arquivos.ufsc.br/d/373be2177a33426a9e6c/)
2. Version 2.1 (will be available soon) - [Mendeley data](https://doi.org/10.17632/wg4bpm33hj.1)

# What is this dataset looks like?
Explanations and examples around the `>=2.0` version of the dataset. If you want to use older versions of the dataset, you will need to make some modifications to the data directory organizations, or things like that.


This is a computer vision dataset, created by some collaborators from different departments at [Universidade Federal de Santa Catarina (UFSC)](https://en.ufsc.br/). The dataset contains images annotated/labelled for semantic segmentation and others. The annotation tool is [labelbox](https://labelbox.com/). In the data repositories will the images, masks (semantic segmentation) and COCO annotations for object detection. The codes to convert annotations from labelbox format to others will be in this package.

Each slide can have some differences in the stain coloration, at figure 1 can be seen an image created from different images of different slides.

![Image sample created from samples from different slides](./data/static_images/Figure1.jpg)

In directory [./data/samples/images/](./data/samples/images/) can be seen the original images of each tile from different slides/patients. The dataset present a wide variety of colors, texture, nuclei format, and others for the cells nuclei, this variety depends on different factors as: Type of lesion, stain process, sample acquisition, sensor/microscopy setup for image acquisition and others.

The dataset at version `1.x` has 3 categories annotated, and at version `2.x` will have 7 categories. But, the principal objective to help at diagnostic/prognostic of these samples is detect/identify/measure the Nucleolus Organizer Regions (NORs) inside each nucleus. The NORs (the black dots/parts inside the nuclei) were labeled as two different categories: Satellite and clusters.

At figure 2, has an example with two highlighted nuclei. The nucleus at left (black highlighted) it's a nucleus with three clusters. The nucleus at right side (gray highlighted) it's a nucleus with one cluster (the black dot at the top of the nuclei) and two satellites (the other two black dots).

![Image from a tile highlighting two nuclei](./data/static_images/Figure2.jpg)

For more explanations about the dataset, see the dataset pages, or their papers.


# Examples of use
TODO
