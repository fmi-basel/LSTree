
# Introduction - Webviewer

The included web-based viewer allows visualizing a lineage tree with a linked view of the 3D cell/nuclei segmentation at a given timepoint.  

<!-- ![webviewer](../docs/webviewer_presentation.mp4) -->

## Running via the notebook / port forwarding
You can run the Webviewer via the jupyter [notebook](webview.ipynb). This is a good first way to look at the tree-meshes visualization, as it is set to look initially at the processed example datasets.

Alternatively, you can also use port forwarding to serve the notebook for remote visualization.

The viewer is then accessible either as a jupyter notebook  or directly served to the browser by activating the lstree environment and running:

```bash
cd /PATH_TO_LSTREE_REPOSITORY/webview
panel serve --static-dirs static_data="static_data" --show webview.ipynb --args --basedir PATH_TO_PROCESSED_DATA
```

Here, `PATH_TO_PROCESSED_DATA` is the full path to where the repository or the datasets folders ( following the structure presented in the [Usage](../README.md) section ).

or on a remote machine:

```bash
cd /PATH_TO_LSTREE_REPOSITORY/webview
panel serve --static-dirs static_data="static_data" --port PORT --allow-websocket-origin=WORKSTATION_NAME:PORT webview.ipynb  --args --basedir PATH_TO_PROCESSED_DATA
```

Example, considering that the data is located in `/Data/LSTree/example/data` and you are connected via e.g. ssh to `workstation1`:

```bash
cd /LSTree/webview
panel serve --static-dirs static_data="static_data" --port 5174 --allow-websocket-origin=workstation1:5174 webview.ipynb  --args --basedir /Data/LSTree/example/data
```

Port forwarding has been extensively tested with `--port=5174`.

Extracted features such as the nuclei volume can be viewed as color:  
<img src="../docs/viewer_volume.png" width="1000"/><br>

Orthogonal views of the original image can be displayed to double check the segmentation:  
<img src="../docs/viewer_stack.png" width="1000"/><br>

It is also possible to highlight certain areas of the tree by manual selection:  
<img src="../docs/viewer_selection.png" width="1000"/><br>

To be responsive, the viewer relies on smoothed 3D meshes pre-generated from the segmentation stacks with:
```bash
LUIGI_CONFIG_PATH=./config.cfg luigi --local-scheduler --module lstree ViewerTask
```



