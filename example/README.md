# Introduction - LSTree example usage

To get acquainted with LSTree, we provide two test datasets that can be used for running the workflow. For simplicity they have already been pre-processed (only cropping, to reduce memory requirements and processing time) and the example lineage trees have been created and exported as MaMuT .xml files. Below we will describe what these steps entail, as well as providing more in-depth information on the main tasks behind the workflow.`

Each new step of the workflow will generate output within a dedicated folder, fwith specific naming conventions. These naming conventions can be changed in the `config.cfg` file.



# Processing steps

## 1. Cropping light-sheet movies
For the recordings of days, organoids may drift inside the matrigel drop and need to be registered back so that the data can be analyzed. In the [crop_movie.ipynb](../notebooks/crop_movie.ipynb) notebook organoids' bounding boxes are first determined on a reference channel and independently for each frame using x,y and z maximum intensity projections (MIPs). Since multiple organoids might appear in the field of view (especially at early time-points), the largest object (or a manually selected object) on the last frame is tracked backward in time by finding its closest match in the previous frame until the first frame is reached. The minimum crop size required for the entire movie is then computed along each axis. At this point crops are reviewed with an included included tool in the notebook and manual corrections can be made, for instance to account for stage movements during medium change. Finally all time-points and channels are cropped by centering the global bounding box on the tracked organoid.

<img src="../docs/cropping_tool.png" width="800"/><br>

## 2. Denoising and deconvolution
Raw images are first denoised with [Noise2Void](https://github.com/juglab/n2v) trained on a few images randomly selected from each movies/channels. The minimum intensity projection along z is used to estimate the background image under the assumption that for each pixel the background is visible on at least one z-slice. Denoised and background-corrected images are then deconvolved with a measured PSF using Richardson-Lucy algorithm running on the GPU using [flowdec](https://github.com/hammerlab/flowdec). This step, including training the denoising model if needed, can be run manually with:

```bash
LUIGI_CONFIG_PATH=./config.cfg luigi --local-scheduler --module lstree MultiDeconvolutionTask
```
> **'-Deconv' folder requirement**: The outputs from this step include a folder with '-Deconv' as a suffix, and the other steps use the images inside as input for segmentation. If this is not needed, i.e. if only raw data is desired to be processed, a - so far still 'hacky' - way around it is to copy the raw images to an empty folder with the same name before running the config file. Keep note that using raw files for processing might require different trainig strategy (e.g. more hand annotations) due to e.g. lower SNR of the images.

> **PSF estimation**: As expected during deconvolution, a point-spread-function (PSF) of the optics used for imaging is necessary. You can uye [Huygens PSF Distiller](https://svi.nl/Huygens-PSF-Distiller) or the [PSF extraction from python-microscopy](http://python-microscopy.org/doc/PSFExtraction.html).

## 3. Lineage tree
LSTree works directly with MaMuT `.xml` files that contain tracking data made using [Mastodon](https://github.com/mastodon-sc/mastodon) Fiji plugin. Subsequently a deep learning model can be (re)trained to predict trees that require fewer manual corrections. Alternatively, one can also use the output from [Elephant](https://elephant-track.github.io/#/v0.3/) ( as MaMuT `.xml` ) to fine tune an existing model or just as final lineage tree output for further processing (segmentation and feature extraction steps). 
To trigger the prediction of lineage trees run:

```bash
LUIGI_CONFIG_PATH=./config.cfg luigi --local-scheduler --module lstree MultiBuildTreeTask
```

The tree prediction outputs include a `.xml` file that can be opened with Mastodon as well as a plots of the predicted tree with overlaid information to facilitate corrections (e.g. nuclei volume, tracking embedding distance). One example of such outputs canbe seen below:

**Tracking prediction outputs**<br>
<img src="../docs/tree_pred.png" width="700"/><br>

## 4. Nuclei segmentation
Nuclei are segmented in 3D following previously reported method: [RDCNet: Instance segmentation with a minimalist recurrent residual network](https://github.com/fmi-basel/RDCNet). A deep learning model model is trained with a mix of complete and partial annotations. Partial labels are obtained by placing a spheres at the position of each tracked nuclei. A small subset of the frames are fully annotated by manually expanding the labels to the full nuclei. Labels can be edited with the provided [notebook](../notebooks/3D_annotator.ipynb). Model architecture and training parameters can be controlled as illustrated in the example configuration file. In particular, initial weights can be supplied with the `resume_weights` option to refine an existing model. To train a model run:

```bash
LUIGI_CONFIG_PATH=./config.cfg luigi --local-scheduler --module lstree NucleiSegmentationTrainingTask
```

At the beginning of the training, a pdf `training_samples.pdf` is exported in the model folder to check that pre-preprocessing steps worked as expected. Training losses can be inspected with Tensorflow TensorBoard `tensorboard --logdir MODEL_FOLDER`. Once the model is trained, nuclei segmentations are predicted using the tracking seeds to enforce the correct number of nuclei and hence temporal consistency. Provided that the .xml lineage tree exist, corresponding nuclei can be segmented with:

```bash
LUIGI_CONFIG_PATH=./config.cfg luigi --local-scheduler --module lstree MultiNucleiSegmentationTask
```

**Important!**

- If no nuclei segmentation exist yet, use the partial annotations generated from the tracked seed as starting point (e.g. nuclei_weak_annot)
    To do this, 
1) Run 
    ```bash
    LUIGI_CONFIG_PATH=./config.cfg luigi --local-scheduler --module lstree NucleiSegmentationTrainingTask
    ```
   to trigger the generation of the partial labels.


2)  After a first prediction has been generated, use the your viewer of choice ( [napari](napari.org), [ImageJ](imagej.net), etc ) to find files for fine tuning of the model. Alternative you can also use the [segmentation viewer](../notebooks/segmentation_viewer.ipynb) to pick and automatically copy files for correction in a separate folder.

## 5. Cell and lumen segmentation
Cell and lumen segmentation also expand on the [RDCNet](https://github.com/fmi-basel/RDCNet) method. The semantic branch predicts 3 classes, background, lumen, epithelium and is supervised by manual annotations of a few frames per datasets. *No manual annotations of individual cells is required.* Instead, the previously segmented nuclei are used as partial annotations under the assumption that they are randomly distributed within the cell compartment. Labels of nuclei belonging to multi-nucleated cells are merged based on the tracking information. Since only the membrane channel is provided as input, the network is forced to learn to segment cells. In practice nuclei are not completely randomly distributed (e.g. corners, tapered elongated cells). We therefore also add a regularization term that encourages voxels in unsupervised regions to vote for one of the instances (without enforcing which one) which yield reasonable cell segmentation in most cases. 

To train a model ( considerirng the existence of nuclei segmentation and lumen annotations ) run:

```bash
LUIGI_CONFIG_PATH=./config.cfg luigi --local-scheduler --module lstree CellSegmentationTrainingTask
```

Lumen segmentation and, if lineage tree exists, cell segmentation can be generated with:
```bash
LUIGI_CONFIG_PATH=./config.cfg luigi --local-scheduler --module lstree MultiCellSegmentationTask
```

## 6. Features extraction
Organoid-level features ( volume, lumen volume, etc ) and cell-level features ( cell/nuclei volume, aspect ratio, distance to lumen, etc ) can be extracted with:
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
<img src="../docs/3D_annotator.png" width="800"/><br>

The segmentation pipeline requires manual annotations of nuclei and lumen/epithelium. The provided [3D_annotator.ipynb](notebooks/3D_annotator.ipynb) can be used to generate 3D segmentation labels. Once an initial segmentation output is generated, it becomes much faster to handpick bad samples and correct them. The [segmentation_viewer.ipynb](notebooks/segmentation_viewer.ipynb) notebook allows quickly inspecting the current segmentation and copying files to be corrected in a separate folder in one click.


**Segmentation viewer**<br>
<img src="../docs/segmentation_viewer.png" width="500"/><br>

---




## Quickstart

### 1) Processing the example data

For an easy first-time use, you can run everything in one go via a single command. This will trigger the entire workflow on one example dataset ( [003-Enterocyst](/example/data/003-Enterocyst/) ) using all models trained with `intestinal organoid images` (this takes around 5 minutes with the hardware configuration we used). The other example dataset ( [002-Budding](/example/data/002-Budding) ) has already been processed for convenience. This means that if you do not have the hardware to run the Task below, you can directly check the output files from there and skip to the [webviewer](../webview/README.md) for visualizing the output.

As long as the data has been cropped and lineage tree created/curated and exported via MaMuT .xml ( which is the case for the example datasets ), the entire LSTree workflow can be activated via the command below (this takes around 10 minutes with the [hardware configuration we used](LINK)):

```bash
LUIGI_CONFIG_PATH=./config.cfg luigi --local-scheduler --module lstree ViewerTask
```

It is also possible to run each step separately on the example data.
If you start from scratch using a different dataset(s) the configuration file needs to be properly changed. More information can be found inside the [config.cfg](../config.cfg) itself as well as in the [configuration file parameters](../config_params.md) readme.
### 2) Visualization of the output

After processing is finished all the outputs can be visualized with the [webviewer](../webview/README.md).
The included web-based viewer allows visualizing a lineage tree with a linked view of the 3D cell/nuclei segmentation at a given timepoint. More information on how to use it, along with example notebook can be found [here](../webview/README.md).




---

> :warning: **Important: How to rerun tasks**: If there are samples for which the output files already exist, then these are skipped. To rerun a specific task* all necessary intermediate and final outputs should be deleted. That also include training deep learning models, i.e. if a trained model exist, it is used without retraining. ( *Lineage tree prediction is currently the only task that can resume in case it is suddenly stopped. ) 



---




