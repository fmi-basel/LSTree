# Utilities

Here you can find a couple of useful tools for using with LSTree. They are presented her in form of jupyter notebooks, and in order to use them you can just activate the `lstree` environment and run 

```bash
jupter notebook --port=XXXX
```
The jupyter notebook should initialize and port forward itself to your localhost using the XXXX specified port. This way you just need to open a browser and type:

```bash
loalhost:XXXX
```
and the jupyter notebook should open. Then you can browser yourself to the /notebook folder where the utilities are located.

## 1. Cropping light-sheet movies
Organoids' bounding boxes are first determined on a reference channel and independently for each frame using x,y and z maximum intensity projections (MIPs). Since multiple organoids might appear in the field of view (especially at early time-points), the largest object (or a manually selected object) on the last frame is tracked backward in time by finding its closest match in the previous frame until the first frame is reached. The minimum crop size required for the entire movie is then computed along each axis. At this point crops are reviewed with the included tool: [crop_movie.ipynb](crop_movie.ipynb) and manual corrections can be made, for instance to account for stage movements during medium change. Finally all time-points and channels are cropped by centering the global bounding box on the tracked organoid.

**3D Annotator**<br>
<img src="../docs/cropping_tool.png" width="800"/><br>

## 2. 3D Annotator

The segmentation pipeline requires manual annotations of nuclei and lumen/epithelium. The provided [3D_annotator.ipynb](3D_annotator.ipynb) can be used to generate 3D segmentation labels. Once an initial segmentation output is generated, it becomes much faster to handpick bad samples and correct them. 

The [segmentation_viewer.ipynb](segmentation_viewer.ipynb) notebook allows quickly inspecting the current segmentation and copying files to be corrected in a separate folder in one click.

**Segmentation viewer**<br>
<img src="../docs/segmentation_viewer.png" width="500"/><br>

Similarly, one can also generate annotations using other tools such as the in-built label layer in [Napari](https://napari.org) and use them for training of networks with LSTree.