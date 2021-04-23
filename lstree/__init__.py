import logging
import luigi
import os

from .features.track_feature_tasks import MultiBuildTreeTask
from .features.agg_feature_tasks import MultiAggregateFeaturesTask
from .features.agg_feature_tasks import MultiAggregateOrganoidFeaturesTask
from .deconv.deconv_tasks import MultiDeconvolutionTask
from .segmentation.nuclei_tasks import NucleiSegmentationTrainingTask
from .segmentation.nuclei_tasks import MultiNucleiSegmentationTask
from .segmentation.cell_tasks import CellSegmentationTrainingTask
from .segmentation.cell_tasks import MultiCellSegmentationTask
from .segmentation.mesh_tasks import ViewerTask

#TODO configure luigi logging??
# setup logging to file. better way to do it with luigi?
logger = logging.getLogger('luigi-interface')
formatter = logging.Formatter(
    '[%(asctime)s] (%(name)s) [%(levelname)s]: %(message)s',
    '%d.%m.%Y %I:%M:%S')
fh = logging.FileHandler(os.path.join(os.getcwd(), 'luigi.log'))
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)
