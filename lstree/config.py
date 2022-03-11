import luigi

SPACING = {25: (2, 0.26, 0.26), 37: (2, 0.173, 0.173)} 

class ExperimentParams(luigi.Config):
    movie_config_name = luigi.Parameter('experiment.json', description='Name of config file (magnification, wavelength, etc.) expected in each movie folder')


class RDCNetParams(luigi.Config):
    input_shape = luigi.TupleParameter((None, None, None, 1), description='model input shape')
    downsampling_factor = luigi.TupleParameter(description='Downsampling factor, can be specified for each dimension separately.')
    n_downsampling_channels = luigi.IntParameter(description='number of channels after downsampling (strided conv).')
    n_groups = luigi.IntParameter(description='number of groups in group conv.')
    channels_per_group = luigi.IntParameter(description='number of channels per group.')
    dilation_rates = luigi.TupleParameter(description='Dilation rates used in dilated conv pyramid')
    dropout = luigi.FloatParameter(0.1, description='spatial dropout rate.')
    n_steps = luigi.IntParameter(5, description='number of steps of delta loop')
    suffix = luigi.Parameter('', description='suffix appended to model name')


class InstanceHeadParams(luigi.Config):
    n_classes = luigi.IntParameter(2, description='number of semantic classes')
    spacing = luigi.TupleParameter((1., ), description='scaling of embedding coordinates, e.g. voxel size.')


class TrainingParams(luigi.Config):
    train_batch_size = luigi.IntParameter(description='training batch size.')
    valid_batch_size = luigi.IntParameter(description='validation batch size.')
    epochs = luigi.IntParameter(description='number of training epochs.')
    learning_rate = luigi.FloatParameter(description='initial learning rate.')
    n_restarts = luigi.IntParameter(1, description='number of restarts for the cosine annealing scheduler')
    patch_size = luigi.TupleParameter(description='training patch size')
    resume_weights = luigi.OptionalParameter(None, description='path to weigths used to resume training')
    plot_dataset = luigi.BoolParameter(True, description='plot samples from from the training set to pdf at the beginning of training', parsing=luigi.BoolParameter.EXPLICIT_PARSING)


class InstanceTrainingParams(luigi.Config):
    intra_margin = luigi.FloatParameter(description='intra embedding margin')
    inter_margin = luigi.FloatParameter(description='inter embedding margin')
    jaccard_hinge = luigi.FloatParameter(0.3, description='lower hinge for binary Jaccard loss')
    jaccard_eps = luigi.FloatParameter(1., description='epsilon/smoothing parameter for binary Jaccard loss')


class AugmentationTrainingParams(luigi.Config):
    noise_mu = luigi.FloatParameter(0., description='mean of the distribution from which sigma is drawn.')
    noise_sigma = luigi.FloatParameter(0., description='standard deviation of the distribution from which sigma is drawn')
    intensity_offset_sigma = luigi.FloatParameter(0., description='standard deviation of the distribution from which sigma is drawn.')
    intensity_scaling_bounds = luigi.TupleParameter((0.1, 10.), description='scaling factor bounds')
