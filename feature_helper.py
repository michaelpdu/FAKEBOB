import os, sys
from os.path import dirname, realpath
root_dir = dirname(dirname(dirname(realpath(__file__))))

fakebob_dir = os.path.join(root_dir, '3rd_party', 'fakebob')
sys.path.append(fakebob_dir)
from ivector_PLDA_kaldiHelper import ivector_PLDA_kaldiHelper
from xvector_PLDA_kaldiHelper import xvector_PLDA_kaldiHelper
from gmm_ubm_kaldiHelper import gmm_ubm_kaldiHelper

utils_dir = os.path.join(root_dir, 'utils')
sys.path.append(utils_dir)
from similarity import calc_similarity

import shutil
import numpy as np
from kaldiio import ReadHelper


class FeatureHelper:
    """    
    """
    def __init__(self, model_type):
        """
        type: support iv|xv|gmm
        """
        self.model_type = model_type
        if model_type == 'iv':
            self.pre_model_dir=os.path.join(fakebob_dir, "kaldi_models", "ivector_models")
        elif model_type == 'xv':
            self.pre_model_dir=os.path.join(fakebob_dir, "kaldi_models", "xvector_models")
        elif model_type == 'gmm':
            raise Exception('Unsupported Model Type!')
            self.pre_model_dir=os.path.join(fakebob_dir, "kaldi_models", "ivector_models")
        else:
            raise Exception('Unsupported Model Type!')

        self.spk_id = os.path.abspath('tmp_spk_id')
        if os.path.exists(self.spk_id):
            shutil.rmtree(self.spk_id)
        os.makedirs(self.spk_id)
        print('Temp Workspace:', self.spk_id)

        self.audio_dir = os.path.abspath(self.spk_id + "/audio")
        self.mfcc_dir = os.path.abspath(self.spk_id + "/mfcc")
        self.log_dir = os.path.abspath(self.spk_id + "/log")
        if model_type == 'iv':
            self.feature_dir = os.path.abspath(self.spk_id + "/ivector")
        elif model_type == 'xv':
            self.feature_dir = os.path.abspath(self.spk_id + "/xvector")
        else: # gmm
            self.feature_dir = os.path.abspath(self.spk_id + "/score")

    def make_tmp_dirs(self):
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.mfcc_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.feature_dir, exist_ok=True)

    def remove_tmp_dirs(self):
        if os.path.exists(self.spk_id):
            shutil.rmtree(self.spk_id)

    def check_path_status(self, audio_path_list):
        """
        return: if bypass verification, return directly, otherwise, raise ERROR
        """
        for audio_path in audio_path_list:
            if not os.path.exists(audio_path):
                raise Exception('File does not exist! '+audio_path)

    def extract(self, audio_path_list, n_jobs=10, debug=False):
        """
        audio_path_list: a list of audio path
        n_jobs: jobs number
        return: ivector (numpy.array)
        """
        # check path status
        self.check_path_status(audio_path_list)

        # prepare temp dirs
        self.make_tmp_dirs()

        # extract ivector in kaldi
        count = len(audio_path_list)
        if count < n_jobs:
            n_jobs = count

        if self.model_type == 'iv':
            helper = ivector_PLDA_kaldiHelper(pre_model_dir=self.pre_model_dir, audio_dir=self.audio_dir, 
                mfcc_dir=self.mfcc_dir, log_dir=self.log_dir, ivector_dir=self.feature_dir)
        elif self.model_type == 'xv':
            helper = xvector_PLDA_kaldiHelper(pre_model_dir=self.pre_model_dir, audio_dir=self.audio_dir, 
                mfcc_dir=self.mfcc_dir, log_dir=self.log_dir, xvector_dir=self.feature_dir)
        else: # gmm
            helper = gmm_ubm_kaldiHelper(pre_model_dir=self.pre_model_dir, audio_dir=self.audio_dir, 
                mfcc_dir=self.mfcc_dir, log_dir=self.log_dir, score_dir=self.feature_dir)

        helper.data_prepare(audio_path_list)
        helper.make_mfcc(n_jobs=n_jobs, debug=debug)
        helper.compute_vad(n_jobs=n_jobs, debug=debug)
        if self.model_type == 'iv':
            helper.extract_ivector(n_jobs=n_jobs, debug=debug)
            ark_fpath = "ark:{}/spk_ivector.ark".format(self.feature_dir)
        elif self.model_type == 'xv':
            helper.extract_xvector(n_jobs=n_jobs, debug=debug)
            ark_fpath = "ark:{}/spk_xvector.ark".format(self.feature_dir)
        else: # gmm
            pass

        # read ivector and normalization
        keys = []
        with ReadHelper(ark_fpath) as reader:
            i = 0
            for key, feature in reader:
                keys.append(key)
                features = np.vstack([feature/np.linalg.norm(feature)]) if i == 0 else np.vstack([features, feature/np.linalg.norm(feature)])
                i+=1

        # remove temp dirs
        self.remove_tmp_dirs()

        return keys, features

if __name__ == '__main__':
    helper = FeatureHelper('xv')
    audio_path_list = [
        os.path.join(fakebob_dir, "data/test-set/61/61-70968-0001.wav"),
        os.path.join(fakebob_dir, "data/test-set/61/61-70968-0031.wav"),
        os.path.join(fakebob_dir, "data/test-set/61/61-70970-0030.wav"),
        os.path.join(fakebob_dir, "data/test-set/1580/1580-141083-0003.wav"),
        os.path.join(fakebob_dir, "data/test-set/2830/2830-3979-0003.wav"),
        os.path.join(fakebob_dir, "data/test-set/4446/4446-2271-0003.wav")
    ]
    keys, features = helper.extract(audio_path_list, debug=False)
    # print(keys)
    # print(features.shape)
    print('Similarity Matrix:')
    print(np.matmul(features, features.T))