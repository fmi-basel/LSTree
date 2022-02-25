# Configuration file parameters

General parameters for each tasks are configured through a global configuration file. For example, deconvolution parameters common to all images can be controlled by:

```
[DeconvolutionTask]
psf_dir=PATH_TO_PSF_IMAGES
out_suffix=-Deconv
niter=128
max_patch_size=(9999,9999,9999)
```

In the configuration file changes in file/folder naming convention, etc, can also be done to adapt to already existing datasets, as long as all datasets follow the same structure.

The configuration file is subdivided into main sections


### **1) Denoising/Deconvolution**
### **2) Lineage**
    Gets all of the spots informations from the existing MaMuT .xml and parses them for basic feature extraction 
### **3) Nuclei segmentation**
    Performs nuclei segmentation model training and prediction tasks on image data based on existing MaMuT .xml tracks.
### **4) Cell segmentation**
    Performs cell and lumen (+ epithelium) segmentation model training and prediction tasks on image data, based on predicted nuclei segmentation.
### **5) Features**
    Extracts features based on the segmented label maps
### **6) Meshes**
    Creates .vtk meshes based on the cell and nuclei segmentation label maps
### **7) Tracking**
    Performs tracking model training and prediction tasks on image data based on existing nuclei segmentation.




> **Important points**: All of the segmentation tasks ultimately depend on existing seeds, i.e. spots from a MaMuT .xml tracking file. Therefore, to start from complete scratch, one should first get tracking data, using either [Mastodon](https://github.com/mastodon-sc/mastodon) or [Elephant](https://elephant-track.github.io/#/v0.3/)
---

## Common tasks
Each of the main sections described above contain one or more luigi-based tasks. For nuclei, cell, and tracking predictions a similar structure is adopted. Considering X = [ *Nuclei*, *Cell*, *Tracking* ]:

- **BuildXTrainingRecordTask** : generates all of the tensorflow records necessary to perform training of a model based on dense annotations. These annotations are searched inside folders called `X_annot`, which should be present inside each experiment folder, and should be `int16` stacks with same naming as the corresponding raw images ( A step-by-step procedure on how to get first annotations done is explained in the [3D_annotator notebook](/notebooks/3D_annotator.ipynb) ). These records are saved as .tfrec files containing the annotation/raw image pairs, and are usually stored inside a models folder. If new datasets are incorporated and a retraining of the model should be done, new .tfrec files should be created and added to the folder. 
<!-- >`training_base_dir`: path to where all `.tfrec` files and models are going to be saved
>`ch_subdir`: the subdirectory where  -->
>`spacing` (z,y,x): dictates the spacing of the generated model afterwards, and is dependent on the spacing of the images used for training.
---

- **BuildWeakXTrainingRecordTask** ( only Nuclei or Cell ) : similar to  **BuildXTrainingRecordTask**, it generates all tfrec files for the `weak annotations`, i.e. sparse annotations of the structure of interest. In the case of nuclei, the weak annotations are the expanded spots from the tracking that still lie inside each nucleous. For cells, these are the corresponding nuclei segmentations which should lie inside each cell.

- **WeakAnnotTask** ( only Nuclei ) : automatically creates the sparse annotations for nuclei training based on expansion of tracking spots from a MaMut.xml file. 
>*Note:* As the name says, the sparse annotations do not cover the entire nucleus in the same way as a dense annotation, however depending on the amount of tracking data, these can be many, which means that in general, "somewhere" inside a nucleous there should be a sparse annotated spot. Since RDCNet is inherently based on a recursive loop, the network can quickly learn through integration of all possible locations that it should segment complete nuclei. This task does not exist for cells, as in there the segmented nuclei are used as sparse annotations, and the training follows without any need to have dense annotations of cells, but with the boundary constraints of where lumen and oranoid should be.
---

- **XSegmentationTrainingTask** : Performs the training itself. Most of its parameters are related to setting the shape of receptive field, as well as RDCNet specific configuration. More information on some of them can be found in the [RDCNet publication](https://arxiv.org/abs/2010.00991). The training can be called from within the lstree environment via

```bash
LUIGI_CONFIG_PATH=./config.cfg luigi --local-scheduler --module lstree MultiAggregateFeaturesTask

```
>`suffix`: this is a string element which is appended to the end of the model name for further reference. If a model with the same suffix already exists, the training does not start.
---
- **XSegmentationTask** : performs the prediction using the trained model.
---
- **MultiXSegmentationTask** ( only Nuclei, Cell ) : performs prediction using the trained model with corresponding `suffix` on all of the datasets present in the list of paths `movie_dirs`. Prediction should be performed on data located in the `ch_subdir` and the ouput written on the defined `out_subdir` folder.

<!-- Add all of the other tasks! -->