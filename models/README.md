# Models

Here you can find all of the models used for nuclei and cells predictions, as well as the lineage tree prediction model.
All of the data used for training of the nuclei segmentation model can be found here  -----zenodo link.


> :warning: All of the models trained here were based on recordings performed with different magnifications (25X and 37.5X), hence also voxel sizes (zyx for 25X: (2.0, 0.26, 0.26) and for 37.5X: (2.0, 0.173, 0.173)). All of the datasets performed with 37.5X were then matched to 25X, meaning that all models are based on anisotropic spacing of (2.0, 0.26, 0.26). If the a new dataset has a different spacing (e.g. isotropic data) then ideally a new model should be trained, so that resolution in Z can be increased.