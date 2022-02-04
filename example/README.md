# Introduction - LSTree example usage

To get acquainted with LSTree, we provide two test datasets that can be used for running the workflow. For simplicity they have already been pre-processed (only cropping, to reduce memory requirements and processing time) and the example lineage trees have been created and exported as MaMuT .xml files. Below we will describe what these steps entail, as well as giving a step-by-step guide on the main tasks behind the workflow.

## Quickstart
As long as the data has been cropped and lineage tree created/curated and exported via MaMuT .xml, the entire LSTree workflow can be activated via

```bash
LUIGI_CONFIG_PATH=./config.cfg luigi --local-scheduler --module lstree ViewerTask
```

By default this command will run on the test dataset provided (002-Budding and 003-Enterocyst) using all models trained with intestinal organoid images. The configuration file must be first adapted to the right input paths before using it on new user data.


---

> :warning: **Important: How to rerun tasks**: If there are samples for which the output files already exist, then these are skipped. To rerun the workflow all necessary intermediate and final outputs should be deleted. That also include training deep learning models, i.e. if a trained model exist, it is used without retraining.

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

<img src="../docs/tree_pred.png" width="700"/><br>

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