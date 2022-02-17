
# Introduction - LSTree

This repository hosts the version of the code used for the [preprint](https://www.biorxiv.org/content/10.1101/2021.05.12.443427v1) *Multiscale light-sheet organoid imaging framework* (de Medeiros, Ortiz et al 2021).

**LSTree** is a digital organoid and lineage tree extraction framework for light sheet movies. It provides pre-processing, analysis and visualization tools which generate multiple features from the 3D recorded data. Ultimately, the extracted features can be visualized onto both lineage trees and 3D segmentation meshes in a combined way via a web-based viewer.

Below you will find instructions on how to install the environment to run LSTree as well as how to run it on the two example datasets.

# Main Sections

## 0) Installation

Main steps for installing LSTree can be found below in the next Section.

## 1) [Example](/example/)

Step-by-step guide of the main functions on two example datasets.

## 2) [Utils](/notebooks/)
A compendium of utilities we initially created to aid on 3D annotations and checking segmentation quality. Also includes the cropping notebook used as a pre-processing step.

## 3) [Models](/models/)

Gathers all of the models trained for segmentation and linega tree prediction used in the manuscript. All of the data used for training can be found here  -----zenodo link.

## 4) [Webviewer](/webview/)

A multiscale visualization tool that aims at bringing lineage trees and segmented data together in one viewer.



# Installation

## Minimum requirements
This workflow was tested on linux machines and is implemented using multiprocessing. It will therefore run faster with more CPUs/GPUs available and requires at least:

- 16 GB of RAM
- TensorFlow compatible GPU with >8 GB of VRAM

The size of the deep learning models might need to be adjusted based on the available VRAM during training.

## Installation steps

It is recommended to create a new python environment and install visualization libraries and cuda (GPU support) with conda:

```bash
conda create -n lstree python=3.7
conda activate lstree
```
Important: currently setup.py uses setuptools in order to install all the packages. Since pytables uses Distutils, we need to install it by hand:

```bash
conda install -c pyviz pytables
```


> :warning: **Cuda GPU support and Tensorflow**: we have tested LSTree using specific version for NVIDIA cudatoolkit (version 10.7) and NVIDIA CUDA® Deep Neural Network library (cuDNN) (version 7) libraries that work with Tensorflow 2.3. This is in accordance to the guidelines from Tensorflow, as specified their [website](https://www.tensorflow.org/install/source#gpu). Therefore it is recommended that you have the right NVIDA driver - according to [wandb.ai](https://wandb.ai/wandb/common-ml-errors/reports/How-to-Correctly-Install-TensorFlow-in-a-GPU-Enabled-Laptop--VmlldzozMDYxMDQ) it should be 418.x or higher. For more information please visit the [tensorflow.org](https://www.tensorflow.org/) website. 


Considering the NVIDIA driver to be already installed, please install the cuda related libraries via:

```bash
conda install cudatoolkit=10.1 cudnn=7
```

Finally we clone this repository and install it onto the new environment:

```bash
git clone https://github.com/fmi-basel/LSTree.git
pip install LSTree/
```

# Usage
The entire analysis pipeline is implemented as a Luigi workflow [https://github.com/spotify/luigi] and majors steps can be run with the commands detailed below and on the following sections. Jupyter notebooks for interactive [visualization of the results](/webview/webview.ipynb) and [drawing 3D labels](/notebooks/3D_annotator.ipynb) are also provided.

## Folder structure
A certain data structure is expected so that the workflow can run smoothly: it should ideally be organized with 2-level sub-folders for movie and channels respectively:

```bash
.
└── MOVIE_DIRECTORY
    ├── experiment.json
    ├── mamut.xml
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

Generated outputs will appear as new sub-folders (E.g. Channel0-Deconv, Channel1-Deconv, nuclei_segmentation, cell_segmentation, etc.). Changes in the details concerning file names can still be cahnged in the [configuration file](config.cfg).

## Expected initial files

Ideally, to be able to extract all features, each movie folder should include a MaMuT (`mamut.xml`) lineage tree (see Lineage tree section below) along with an [experiment.json](/example/data/002-Budding/experiment.json) file containing information about acquisition settings which are used e.g. for rescaling, deconvolution and for showing the data with the right temporal spacing, among others:

```
{
    "mag": 25,
    "time_interval": 0.1667,
    "spacing": [
        2,
        0.26,
        0.26
    ],
    "wavelengths": {
        "Channel0": 488,
        "Channel1": 561,
        "Channel2": 638
    }
}
```  


## Configuration file
General parameters for each tasks are configured through a global configuration file [config.cfg](config.cfg). For example, deconvolution parameters common to all images can be controlled by:

```
[DeconvolutionTask]
psf_dir=PATH_TO_PSF_IMAGES
out_suffix=-Deconv
niter=128
max_patch_size=(9999,9999,9999)
```

In the configuration file changes in file/folder naming convention, etc, can also be done to adapt to already existing datasets, as long as all datasets follow the same structure.

More information is discussed within the [example guide](example/README.md).

## Quickstart
Acquired movie should first be cropped (see below) to reduce memory requirements and processing time. Then, provided that initial nuclei and lumen annotations, as well as lineage tree (generated/curated with Mastodon) are available, the entire pipeline can be triggered with:

```bash
LUIGI_CONFIG_PATH=./config.cfg luigi --local-scheduler --module lstree ViewerTask
```


By default this command will run on the test dataset provided (003-20170313-P4-25X-Enterocyst) using all models trained with intestinal organoid images. The configuration file must be first adapted to the right input paths before using it on new user data.


---
**NOTE**

If there are samples for which the output files already exist, then these are skipped. To rerun the workflow all necessary intermediate and final outputs should be deleted. That also include training deep learning models, i.e. if a trained model exist, it is used without retraining.

---






# Processing steps

## 1. Cropping lights-heet movies
Organoids' bounding boxes are first determined on a reference channel and independently for each frame using x,y and z maximum intensity projections (MIPs). Since multiple organoids might appear in the field of view (especially at early time-points), the largest object (or a manually selected object) on the last frame is tracked backward in time by finding its closest match in the previous frame until the first frame is reached. The minimum crop size required for the entire movie is then computed along each axis. At this point crops are reviewed with the included tool: [crop_movie.ipynb](notebooks/crop_movie.ipynb) and manual corrections can be made, for instance to account for stage movements during medium change. Finally all time-points and channels are cropped by centering the global bounding box on the tracked organoid.

<img src="docs/cropping_tool.png" width="800"/><br>

## 2. Denoising and deconvolution
Raw images are first denoised with a model trained with the [Noise2Void](https://github.com/juglab/n2v) scheme on a few images randomly selected from each movies/channels. The minimum intensity projection along z is used to estimate the background image under the assumption that for each pixel the background is visible on at least one z-slice. Denoised and background-corrected images are then deconvolved with a measured PSF using Richardson-Lucy algorithm running on the GPU. This step, including training the denoising model if needed, can be run manually with:

```bash
LUIGI_CONFIG_PATH=./config.cfg luigi --local-scheduler --module lstree MultiDeconvolutionTask
```

## 3. Lineage tree
Initially nuclei have to be tracked manually using [Mastodon](https://github.com/mastodon-sc/mastodon) Fiji plugin. Subsequently a deep learning model can be (re)trained to predict trees that require fewer manual corrections.

```bash
LUIGI_CONFIG_PATH=./config.cfg luigi --local-scheduler --module lstree MultiBuildTreeTask
```

The tree prediction outputs include a `.xml` file that can be opened with Mastodon as well as a plots of the predicted tree with overlaid information to facilitate corrections (e.g. nuclei volume, tracking embedding distance).

<img src="docs/tree_pred.png" width="700"/><br>

## 4. Nuclei segmentation
Nuclei are segmented in 3D following previously reported method: [RDCNet: Instance segmentation with a minimalist recurrent residual network](https://github.com/fmi-basel/RDCNet). A deep learning model model is trained with a mix of complete and partial annotations. Partial labels are obtained by placing a spheres at the position of each tracked nuclei. A small subset of the frames are fully annotated by manually expanding the labels to the full nuclei. Labels can be edited with the provided [notebook](notebooks/3D_annotator.ipynb). Model architecture and training parameters can be controlled as illustrated in the example configuration file. In particular, initial weights can be supplied with the `resume_weights` option to refine an existing model. To train a model run:

```bash
LUIGI_CONFIG_PATH=./config.cfg luigi --local-scheduler --module lstree NucleiSegmentationTrainingTask
```

At the beginning of the training, a pdf `training_samples.pdf` is exported in the model folder to check that pre-preprocessing steps worked as expected. Training losses can be inspected with Tensorflow TensorBoard `tensorboard --logdir MODEL_FOLDER`. Once the model is trained, nuclei segmentations are predicted using the tracking seeds to enforce the correct number of nuclei and hence temporal consistency. Provided that the .xml lineage tree exist, corresponding nuclei can be segmented with:

```bash
LUIGI_CONFIG_PATH=./config.cfg luigi --local-scheduler --module lstree MultiNucleiSegmentationTask
```

## 5. Cell and lumen segmentation
Cell and lumen segmentation also expand on the [RDCNet](https://github.com/fmi-basel/RDCNet) method. The semantic branch predicts 3 classes, background, lumen, epithelium and is supervised by manual annotations of a few frames per datasets. No manual annotations of individual cells is required. Instead the previously segmented nuclei are used as partial annotations under the assumption that they are randomly distributed within the cell compartment. Labels of nuclei belonging to multi-nucleated cells are merged based on the tracking information. Since only the membrane channel is provided as input, the network is forced to learn to segment cells. In practice nuclei are not completely randomly distributed (e.g. corners, tapered elongated cells). We therefore also add a regularization term that encourages voxels in unsupervised regions to vote for one of the instances (without enforcing which one) which yield reasonable cell segmentation in most cases. To train a model run:

```bash
LUIGI_CONFIG_PATH=./config.cfg luigi --local-scheduler --module lstree CellSegmentationTrainingTask
```

Lumen segmentation and, if lineage tree exists, cell segmentation can be generated with:
```bash
LUIGI_CONFIG_PATH=./config.cfg luigi --local-scheduler --module lstree MultiCellSegmentationTask
```

## 6. Features extraction
Organoid-level features (volume, lumen volume, etc.) and cell-level features (cell/nuclei volume, aspect ratio, distance to lumen, etc.) can be extracted with:
```bash
LUIGI_CONFIG_PATH=./config.cfg luigi --local-scheduler --module lstree MultiAggregateFeaturesTask
```
or only organoid-level features if the lineage tree is not available with:
```bash
LUIGI_CONFIG_PATH=./config.cfg luigi --local-scheduler --module lstree MultiAggregateOrganoidFeaturesTask
```

It creates a *.csv file containing the extracted features for each timepoint as well as final a *.hdf file with all time-points aggregated. The *.hdf file contains Pandas DataFrames of nodes and edges with the associated features which can be used to plot lineage trees as illustrated in [plot_features.ipynb](notebooks/plot_features.ipynb).

Alternatively, a generic set of features can be extracted from all combination of intensity images and labels. This can be useful to use externally generated labels. To replace the default feature extraction task, add and adapt the following to the configuration file:

```
[GenericExtractFeaturesTask]
out_subdir=features
label_subdirs=["nuclei_segmentation", "nuclei_surround"]
raw_channel_subdirs=["Channel01", "Channel02"]

[AggregateFeaturesTask]
extractor_type=generic
```

---
**NOTE - Generating training labels**


**3D annotator**<br>
<img src="docs/3D_annotator.png" width="800"/><br>

The segmentation pipeline requires manual annotations of nuclei and lumen/epithelium. The provided [3D_annotator.ipynb](notebooks/3D_annotator.ipynb) can be used to generate 3D segmentation labels. Once an initial segmentation output is generated, it becomes much faster to handpick bad samples and correct them. The [segmentation_viewer.ipynb](notebooks/segmentation_viewer.ipynb) notebook allows quickly inspecting the current segmentation and copying files to be corrected in a separate folder in one click.


**Segmentation viewer**<br>
<img src="docs/segmentation_viewer.png" width="500"/><br>

---



## 7. Visualization
The included web-based viewer allows visualizing a lineage tree with a linked view of the 3D cell/nuclei segmentation at a given timepoint. More information on how to use it, along with example notebook can be found [here](webview/README.md).





## Implementation details
- The workflow and its tasks'dependencies are managed using [Luigi](https://github.com/spotify/luigi).
- Processing steps are batched per movie/channel to amortize tensorflow model initialization.
- Due to current tensorflow limitations, tensorflow should not be imported (directly or indirectly) in the main process but in the `run()` function of each Luigi task (i.e. in sub-processes spawned by Luigi)
- Negative labels in training annotations are considered "not labeled" and do not contribute to the training loss (partial annotations)



# Funding support
This work was supported by EMBO (ALTF 571-2018 to G.M.), SNSF (POOP3_157531 to P.L.). This work received funding from the ERC under the European Union’s Horizon 2020 research and innovation programme (grant agreement no. 758617).
