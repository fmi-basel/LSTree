import abc
import luigi
import os
import numpy as np
import random
import hashlib
import json
import concurrent

from tqdm import tqdm
from multiprocessing import Lock, Manager


def get_task_hash(task):
    params = task.to_str_params()
    param_str = json.dumps(task.to_str_params(),
                           separators=(',', ':'),
                           sort_keys=True)
    return hashlib.md5(param_str.encode('utf-8')).hexdigest()


def stable_hash(a_string):
    '''Returns a hash that does not change with python instance (i.e. unlike hash()) '''

    x = a_string.encode('utf-8')
    return int.from_bytes(hashlib.sha256(x).digest()[:8],
                          byteorder='big',
                          signed=False)


def thread_lock_wrapper(fun, lock):
    def _wrapped_fun(*args, **kargs):
        with lock:
            return fun(*args, **kargs)

    return _wrapped_fun


def monitor_futures(futures, pbar=False):
    '''Collect futures as completed and cancel on first encountered error'''

    #NOTE if progress bar was no needed, one could use:
    # done_futures, not_done_futures = concurrent.futures.wait(futures, return_when='FIRST_EXCEPTION')
    # ...

    iterator = concurrent.futures.as_completed(futures)
    if pbar:
        iterator = tqdm(iterator, total=len(futures))

    for f in iterator:
        if f.exception() is not None:
            [f.cancel() for f in futures]
            break

    return [f.result() for f in futures]


class ExternalInputFile(luigi.ExternalTask):
    path = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.path)


class BuildTrainingRecordBaseTask(luigi.Task):
    '''Base task to build train/valid/test records
    '''

    training_base_dir = luigi.Parameter(
        description='Directory to save the training records')
    record_name = luigi.Parameter()

    train_fraction = luigi.FloatParameter(default=0.8)
    valid_fraction = luigi.FloatParameter(default=0.1)
    seed = luigi.IntParameter()

    def output(self):
        targets = {}
        for split_name in ['train', 'valid', 'test']:
            outpath = os.path.join(
                self.training_base_dir,
                '{}_{}.tfrec'.format(self.record_name, split_name))
            targets[split_name] = luigi.LocalTarget(outpath)

        return targets

    @abc.abstractmethod
    def _data_gen(self, inputs):
        '''generator to build the training record'''
        pass

    @abc.abstractmethod
    def _get_serialization_fun(self):
        '''Returns a record parser serialization function'''
        pass

    def _split_data(self):
        '''split annot/raw input pairs into train|valid|test sets'''

        if not 0. <= self.train_fraction <= 1.:
            raise ValueError(
                'training fraction should be within [0.,1.]. got {}'.format(
                    self.train_fraction))
        if not 0. <= self.valid_fraction <= 1.:
            raise ValueError(
                'training fraction should be within [0.,1.]. got {}'.format(
                    self.valid_fraction))

        if hasattr(self, 'record_input'):
            record_input = self.record_input
        else:
            record_input = self.input()

        # randomize sample order, e.g mixes timepoints
        record_input = list(record_input)
        random.seed(self.seed)
        random.shuffle(record_input)

        test_fraction = max(0., 1. - self.train_fraction - self.valid_fraction)
        n_samples = len(record_input)
        probs = [self.train_fraction, self.valid_fraction, test_fraction]
        split = np.random.choice([0, 1, 2],
                                 size=n_samples,
                                 replace=True,
                                 p=probs)
        sets = {}
        for idx, split_name in enumerate(['train', 'valid', 'test']):
            inputs = [p for s, p in zip(split, record_input) if s == idx]
            sets[split_name] = inputs

        return sets

    def run(self):
        from dlutils.dataset.tfrecords import tfrecord_from_iterable

        np.random.seed(self.seed)

        for split_name, inputs in self._split_data().items():

            with self.output()[split_name].temporary_path(
            ) as self.temp_output_path:
                tfrecord_from_iterable(self.temp_output_path,
                                       self._data_gen(inputs),
                                       self._get_serialization_fun(),
                                       verbose=False)


# NOTE alternatively check actual usage with nvidia-smi, (e.g. nvgpu python wrapper)
# (tensorflow would have to be initialized and allocate memory before releasing lock)
class GPUTask(luigi.Task):

    _lock = Lock()
    _used_gpus = Manager().dict()

    resources = {'gpu': 1}

    def _acquire_gpu(self):

        import tensorflow as tf
        physical_devices = tf.config.list_physical_devices('GPU')

        self.gpu_idx = -1
        gpu_device = []
        with GPUTask._lock:
            for idx, device in enumerate(physical_devices):
                if not GPUTask._used_gpus.get(idx, False):
                    GPUTask._used_gpus[idx] = True
                    self.gpu_idx = idx
                    gpu_device = [device]
                    break

        if self.gpu_idx < 0:
            raise RuntimeError(
                'no available GPU found. Check that luigi resources "gpu" matches the number of physical GPU'
            )
            # TODO try get "gpu" from luigi config and compare to number of available gpu
            # log warning instead and attempt to run on cpu?

        # print('Placing on GPU {}'.format(self.gpu_idx))
        tf.config.set_visible_devices(gpu_device, 'GPU')

        # to be able to estimate VRAM usage with nvidia-smi
        tf.config.experimental.set_memory_growth(
            physical_devices[self.gpu_idx], True)

    def _release_gpu(self):
        if hasattr(self, 'gpu_idx'):
            with GPUTask._lock:
                GPUTask._used_gpus[self.gpu_idx] = False

    def run(self):
        self._acquire_gpu()
        self.gpu_run()
        self._release_gpu()

    def on_failure(self, exception):
        self._release_gpu()
        return super().on_failure(exception)

    @abc.abstractmethod
    def gpu_run(self):
        pass


def _format_tuple(val):
    '''Format tuple param to write in model name'''

    unique_val = tuple(set(val))

    if len(unique_val) == 1:
        return str(unique_val[0])
    else:
        return str(val).replace(', ', '-').replace('(', '').replace(')', '')
