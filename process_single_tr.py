import nibabel as nib
import logging
import numpy as np
import math
from scipy.stats.mstats import zscore
from brainiak.fcma.classifier import Classifier
from sklearn.externals import joblib

logger = logging.getLogger(__name__)

class SingleTRProcessor:
    def __init__(self, epoch_file, mask_file, model_file, total, window):
        self.epoch = np.load(epoch_file)
        self.mask = nib.load(mask_file).get_data()
        self.total = total
        self.window = window
        self.num_voxels = len(np.where(self.mask > 0)[0])
        self.raw_data = np.zeros((self.total, self.num_voxels), np.float32, order='C')
        self.current_epoch = -1
        self.clf = joblib.load(model_file)
        assert isinstance(self.clf, Classifier), \
                'the loaded classifier is incorrect'
        logger.debug(
                '%d voxels per brain in the classifier, ' \
                '%d training samples involved' %
                (self.clf.num_voxels, self.clf.num_samples)
                )
        logger.info(
                'Single TR processor initialized, '\
                'ready to perform the real-time classification'
                )

    def process_single_tr_nifti(self, nifti_file, tr_count):
        # read
        data = nib.load(nifti_file).get_data()
        # apply mask and accumulate data
        self.raw_data[tr_count, :] = np.copy(data[self.mask > 0])
        # compute correlation if needed
        if self.current_epoch == -1:
            if self.epoch[tr_count] == 1:
                self.current_epoch = tr_count
        elif self.epoch[tr_count] == 0:
            self.current_epoch = -1
        elif tr_count - self.current_epoch + 1 >= self.window:
            # normalize the raw data
            raw_data = self.raw_data[self.current_epoch: tr_count + 1]
            raw_data = zscore(raw_data, axis=0, ddof=0)
            # if zscore fails (standard deviation is zero),
            # set all values to be zero
            raw_data = np.nan_to_num(raw_data)
            raw_data = raw_data / math.sqrt(raw_data.shape[0])
            # predict, waiting for BrainIAK
            y_pred = self.clf.predict([raw_data])
            logger.info('predicted: %d', y_pred[0])
        return
