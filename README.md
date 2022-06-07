<img src="./docs/Repo_visual_outline2.png" width="700"/><br>


# Introduction - LSTree

This repository hosts the version of the code used for the [preprint](https://www.biorxiv.org/content/10.1101/2021.05.12.443427v1) *Multiscale light-sheet organoid imaging framework* (de Medeiros, Ortiz et al 2021).

**LSTree** is a digital organoid and lineage tree extraction workflow for light-sheet movies. It combines pre-processing and analysis tools to extract multiple features from the 3D recorded data. Ultimately, the extracted features can be visualized onto both lineage trees and 3D segmentation meshes in a combined way via a web-based viewer.

Below you will find instructions on how to install the environment to run LSTree as well as how to run it on the two example datasets.

# Table of Contents

###  [Installation](#Installation)

###  [Usage](#Usage)

General information on the way LSTree functions.

### [Example](/example/)

Step-by-step guide of the main functions on two example datasets.

### [Utils](/notebooks/)
A compendium of utilities we initially created to aid on 3D annotations and checking segmentation quality. Also includes a cropping tool which can be used as a pre-processing step. These utilities are all written in form of jupyter notebooks.

### [Models](/models/)

Gathers all of the models trained for segmentation and linega tree prediction used in the manuscript. 

### [Webviewer](/webview/)

A multiscale visualization tool that aims at bringing lineage trees and segmented data together in one viewer.



# Installation

## Minimum requirements
This workflow was tested on linux machines and is implemented using multiprocessing. It will therefore run faster with more CPUs/GPUs available and requires at least:

- 16 GB of RAM
- TensorFlow compatible GPU with >8 GB of VRAM

Currently the tasks have been tested with a16 core Intel Xeon W-2145, 64 GB 2666MHz DDR4 RAM equipped with a Nvidia Quadro RTX 6000 GPU with 24 GB VRAM and using Ubuntu 18.04.6 LTS. 

The size of the deep learning models might need to be adjusted based on the available VRAM during training.

## Installation steps

It is recommended to create a new python environment and install visualization libraries and cuda (GPU support) with conda. For more information on how to install and use (mini)conda please visit the [official conda documentation](https://docs.conda.io/en/latest/miniconda.html). First create the new environment:

```bash
conda create -n lstree python=3.7
```
Activate is via:
```bash
conda activate lstree
```
Important: currently setup.py uses setuptools in order to install all the packages. Since pytables uses Distutils, we need to install it by hand:

```bash
conda install -c pyviz pytables==3.6.1
```



> :warning: **Cuda GPU support and Tensorflow**: we have tested LSTree using specific version for NVIDIA cudatoolkit (version 10.1) and NVIDIA CUDA® Deep Neural Network library (cuDNN) (version 7.6) libraries that work with Tensorflow 2.3. This is in accordance to the guidelines from Tensorflow, as specified their [website](https://www.tensorflow.org/install/source#gpu). Therefore it is recommended that you have the right NVIDIA driver - according [NVIDIA website](https://docs.nvidia.com/deploy/cuda-compatibility/) this should be equal or greater than 418.39. For more information please visit the [tensorflow.org](https://www.tensorflow.org/) website. 
Considering the NVIDIA driver to be already installed, please install the cuda related libraries via:


```bash
conda install cudatoolkit=10.1 cudnn=7
```
To run the notebooks present as example and utils here, you also need `jupyter notebook` installed:
```bash
conda install jupyter notebook
```

Finally we clone this repository and install it onto the new environment. Be aware that currently the example datasets are included in the repository and therefore cloning will require around 1 GB of storage space. Furthermore, cloning this repository will also install [inter-view](https://github.com/fmi-basel/inter-view), [dl-utils](https://github.com/fmi-basel/dl-utils) and [improc](https://github.com/fmi-basel/improc), as LSTree utilizes many functionalities from them.

```bash
git clone https://github.com/fmi-basel/LSTree.git
```

```bash
pip install LSTree/
```

# Usage
The entire analysis pipeline is implemented as a [Luigi workflow](https://github.com/spotify/luigi) and majors steps can be run with the commands detailed below and on the following sections. Jupyter notebooks for interactive [visualization of the results](/webview/webview.ipynb) and [drawing 3D labels](/notebooks/3D_annotator.ipynb) are also provided.



## Initial Requirements 

### **MaMuT.xml**

As the name hints, **LSTree** heavily depends on the existence of tracking data, in form of a MaMuT `.xml` file, so that all features of a light-sheet recording can be extracted. To fulfill this requirement lineage trees can be created via [Mastodon](https://github.com/mastodon-sc/mastodon), or [Elephant](https://elephant-track.github.io/#/v0.3/) or any other tracking algorithm as long as the output can be written in MaMuT `.xml` form.Ideally, to be able to extract all features, each movie folder should include a MaMuT (`mamut.xml`) lineage tree (see Lineage tree section below) along with an [experiment.json](/example/data/002-Budding/experiment.json) file containing information about acquisition settings which are used e.g. for rescaling, deconvolution and for showing the data with the right temporal spacing, among others:


### **experiment.json**
En [experiment.json](/example/data/002-Budding/experiment.json) file containing information about acquisition settings which are used e.g. for rescaling, deconvolution and for showing the data with the right temporal spacing, among others:

```
{
    "mag": 25,                  # magnification (for getting the right PSF file)
    "time_interval": 0.1667,    # value in hours
    "spacing": [                # values in µm
        2,                      # Z
        0.26,                   # Y
        0.26                    # X
    ],
    "wavelengths": {            # values in nm
        "Channel0": 488,        
        "Channel1": 561,
        "Channel2": 638
    }
}
```  
### **dataset.xml**

Before performing segmentation and feature extraction, LSTree requires a `dataset.xml` file that has the structure of the BigDataViewer `.xml`.
In the two [example datasets](/example/README.md) we have added the `.xml ` files correspondingly. For mor information on its structure, please refer to [BigDataViewer](https://imagej.net/plugins/bdv/).


## Folder structure
A certain data structure is expected so that the workflow can run smoothly: it should ideally be organized with 2-level sub-folders for movie and channels respectively:

```bash
.
└── MOVIE_DIRECTORY
    ├── experiment.json
    ├── mamut.xml
    ├── dataset.xml
    ├── nuclei_annot
    │   ├── FILENAME-T0017.tif
    │   ├── FILENAME-T0134.tif
    │   └── FILENAME-T0428.tif
    ├── lumen_annot
    │   ├── FILENAME-T0024.tif
    │   ├── FILENAME-T0245.tif
    │   └── FILENAME-T0712.tif
    ├── Channel0
    │   ├── FILENAME-T0001.tif
    │   ├── FILENAME-T0002.tif
    │   ├── FILENAME-T0003.tif
    │   ├── FILENAME-T0004.tif
    │   ├── FILENAME-T0005.tif
    │   .
    │   └── FILENAME-Tnnnn.tif
    └── Channel1
        ├── FILENAME-T0001.tif
        ├── FILENAME-T0002.tif
        ├── FILENAME-T0003.tif
        ├── FILENAME-T0004.tif
        ├── FILENAME-T0005.tif
        .
        └── FILENAME-Tnnnn.tif
 ```

Generated outputs will appear as new sub-folders (E.g. Channel0-Deconv, Channel1-Deconv, nuclei_segmentation, cell_segmentation, etc.).

For more information on the way this works in practice, please visit the [example](/example/README.md) section!

## Configuration file
General parameters for each tasks are configured through a global configuration file [config.cfg](config.cfg), which is initially set up to run the [example](example/README.md) datasets.



---




## Implementation details
- The workflow and its tasks' dependencies are managed using [Luigi](https://github.com/spotify/luigi).
- Processing steps are batched per movie/channel to amortize tensorflow model initialization.
- Due to current tensorflow limitations, tensorflow should not be imported (directly or indirectly) in the main process but in the `run()` function of each Luigi task (i.e. in sub-processes spawned by Luigi)
- Negative labels in training annotations are considered "not labeled" and do not contribute to the training loss (partial annotations)


---
## Acknowledgements

Big thanks to the [Liberali Lab](https://liberalilab.org/) for enabling the creation of these tools, the support and help with the initial development, discussions and future plans of LSTree. Also to Ko Sugawara for the nice collaboration regarding the utilization of `Elephant` and its integration within the `LSTree` umbrella. Thanks to Vladimir Ulman for helpful discussions regarding tracking in general, and Markus Rempfler for important initial support with RDCNet implementations. Also big thanks to Tim-Oliver for insightful discussions and help with debugging. 

---
## Funding support
This work was supported by EMBO (ALTF 571-2018 to G.M.), SNSF (POOP3_157531 to P.L.). This work received funding from the ERC under the European Union’s Horizon 2020 research and innovation programme (grant agreement no. 758617).

---
## How to cite

If LSTree is being useful to your analysis / visualization, please do not forget to cite it using the citation file.


