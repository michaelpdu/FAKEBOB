from gmm_ubm_kaldiHelper import gmm_ubm_kaldiHelper
from ivector_PLDA_kaldiHelper import ivector_PLDA_kaldiHelper
from xvector_PLDA_kaldiHelper import xvector_PLDA_kaldiHelper

import os
import numpy as np
import subprocess
import shlex
import pickle
import shutil
import argparse

class SpeakerModelBuilder:
    '''
    '''

    def __init__(self,enroll_dir,illegal_dir,test_dir,znorm_dir,n_jobs):
        ''' adjustable setting
        '''
        self.n_jobs = n_jobs
        self.debug = True # whether display log information from kaldi on terminal
        self.enroll_dir = enroll_dir # "./data/enrollment-set" # voice data for enrollment
        self.z_norm_dir = znorm_dir # "./data/z-norm-set" # voice data for z norm
        self.test_dir = test_dir # "./data/test-set" # used for setting threshold
        self.illegal_dir = illegal_dir # "./data/illegal-set" # used for setting threshold
        self.ivector_model_dir = "./kaldi_models/ivector_models"
        self.gmm_model_dir = "./kaldi_models/gmm_models"
        self.xvector_model_dir = "./kaldi_models/xvector_models"
        self.model_dir = "./tmpfs/model"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def load_data(self):
        ''' Prepare data for building...
        '''
        audio_iter = os.listdir(self.enroll_dir)
        self.enroll_utt_id = []
        self.enroll_spk_id = []
        self.enroll_utt_path = []
        for i, audio_name in enumerate(audio_iter):
            utt_id = audio_name.split(".")[0]
            spk_id = utt_id.split("-")[0]
            path = os.path.join(self.enroll_dir, audio_name)
            self.enroll_utt_path.append(path)
            self.enroll_utt_id.append(utt_id)
            self.enroll_spk_id.append(spk_id)

        audio_iter = os.listdir(self.z_norm_dir)
        self.z_norm_utt_id = []
        self.z_norm_spk_id = []
        self.z_norm_utt_path = []
        for i, audio_name in enumerate(audio_iter):
            utt_id = audio_name.split(".")[0]
            spk_id = utt_id.split("-")[0]
            path = os.path.join(self.z_norm_dir, audio_name)
            self.z_norm_utt_path.append(path)
            self.z_norm_utt_id.append(utt_id)
            self.z_norm_spk_id.append(spk_id)

        spk_iter = os.listdir(self.test_dir)
        self.test_utt_id = []
        self.test_spk_id = []
        self.test_utt_path = []
        for spk_id in spk_iter:
            spk_dir = os.path.join(self.test_dir, spk_id)
            audio_iter = os.listdir(spk_dir)
            for i, audio_name in enumerate(audio_iter):
                utt_id = audio_name.split(".")[0]
                path = os.path.join(spk_dir, audio_name)
                self.test_utt_path.append(path)
                self.test_utt_id.append(utt_id)
                self.test_spk_id.append(spk_id)

        spk_iter = os.listdir(self.illegal_dir)
        self.illegal_utt_id = []
        self.illegal_spk_id = []
        self.illegal_utt_path = []
        for spk_id in spk_iter:
            spk_dir = os.path.join(self.illegal_dir, spk_id)
            audio_iter = os.listdir(spk_dir)
            for i, audio_name in enumerate(audio_iter):
                utt_id = audio_name.split(".")[0]
                path = os.path.join(spk_dir, audio_name)
                self.illegal_utt_path.append(path)
                self.illegal_utt_id.append(utt_id)
                self.illegal_spk_id.append(spk_id)

        self.audio_path_list = (self.enroll_utt_path + self.z_norm_utt_path + self.test_utt_path + self.illegal_utt_path)
        self.spk_id_list = (self.enroll_spk_id + self.z_norm_spk_id + self.test_spk_id + self.illegal_spk_id)
        self.utt_id_list = (self.enroll_utt_id + self.z_norm_utt_id + self.test_utt_id + self.illegal_utt_id)

    def set_threshold(self, score_target, score_untarget):
        if not isinstance(score_target, np.ndarray):
            score_target = np.array(score_target)
        if not isinstance(score_untarget, np.ndarray):
            score_untarget = np.array(score_untarget)

        minimal_n = np.minimum(score_target.size, score_untarget.size)
        score_target = np.random.choice(score_target, minimal_n, replace=False) if score_target.size > minimal_n else score_target
        score_untarget = np.random.choice(score_untarget, minimal_n, replace=False) if score_untarget.size > minimal_n else score_untarget
        score_target = np.sort(score_target)

        final_threshold = 0.
        min_difference = np.infty
        final_far = 0.
        final_frr = 0.
        for i, candidate_threshold in enumerate(score_target):
            frr = i * 100 / minimal_n
            far = (np.argwhere(score_untarget >= candidate_threshold).flatten().size) * 100 / minimal_n
            difference = np.abs(frr - far)
            if difference < min_difference:

                final_threshold = candidate_threshold
                final_far = far
                final_frr = frr
                min_difference = difference
        return final_threshold, final_frr, final_far

    def build_gmm_model(self):
        audio_dir = os.path.abspath("tmpfs/build_spk_models/iv-audio")
        if not os.path.exists(audio_dir):
            print('ERROR: Cannot find ', audio_dir)
            exit(-1)
        feats_scp = audio_dir + "/feats.scp"
        vad_scp = audio_dir + "/vad.scp"

        audio_dir_gmm = os.path.abspath("tmpfs/build_spk_models/gmm-audio")
        if os.path.exists(audio_dir_gmm):
            shutil.rmtree(audio_dir_gmm)
        os.makedirs(audio_dir_gmm)
        mfcc_dir_gmm = os.path.abspath("tmpfs/build_spk_models/gmm-mfcc")
        if os.path.exists(mfcc_dir_gmm):
            shutil.rmtree(mfcc_dir_gmm)
        os.makedirs(mfcc_dir_gmm)
        log_dir_gmm = os.path.abspath("tmpfs/build_spk_models/gmm-log")
        if os.path.exists(log_dir_gmm):
            shutil.rmtree(log_dir_gmm)
        os.makedirs(log_dir_gmm)
        score_dir = os.path.abspath("tmpfs/build_spk_models/gmm-score")
        if os.path.exists(score_dir):
            shutil.rmtree(score_dir)
        os.makedirs(score_dir)

        #
        dubm = os.path.abspath(os.path.join(self.gmm_model_dir, "final.dubm"))
        delta_opts_file = os.path.join(self.gmm_model_dir, "delta_opts")
        with open(delta_opts_file, "r") as reader:
            delta_opts = reader.read()[:-1]
        update_flags_str = "m" # only update the mean vectors of gmm

        print("--- obtaining gmm identity by updating ubm via MAP ---")
        tmp_spk_feats_scp = audio_dir + "/feats_spk.scp"
        tmp_spk_vad_scp = audio_dir + "/vad_spk.scp"
        tmp_spk_acc_file = audio_dir + "/gmm_map_acc.acc"

        feats_utt_location = np.loadtxt(feats_scp, dtype=str)
        feats_utt = feats_utt_location[:, 0]
        feats_location = feats_utt_location[:, 1]
        vad_utt_location = np.loadtxt(vad_scp, dtype=str)
        vad_utt = vad_utt_location[:, 0]
        vad_location = vad_utt_location[:, 1]

        for spk_id, utt_id in zip(self.enroll_spk_id, self.enroll_utt_id):
            # 
            index = np.argwhere(feats_utt == utt_id).flatten()[0]
            location = feats_location[index]
            spk_feats_scp_content = utt_id + " " + location + "\n"
            with open(tmp_spk_feats_scp, "w") as writer:
                writer.write(spk_feats_scp_content)
            
            index = np.argwhere(vad_utt == utt_id).flatten()[0]
            location = vad_location[index]
            spk_vad_scp_content = utt_id + " " + location +  "\n"
            with open(tmp_spk_vad_scp, "w") as writer:
                writer.write(spk_vad_scp_content)
            
            # 
            add_deltas = ("add-deltas " + delta_opts + " scp:" + tmp_spk_feats_scp + " ark:- |")
            apply_cmvn = "apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- |"
            select_voiced_frame = ("select-voiced-frames ark:- scp,s,cs:"  + tmp_spk_vad_scp + " ark:- |")
            feats = ("ark,s,cs:" + add_deltas + " " + apply_cmvn + " " + select_voiced_frame)
            acc_stats_command = ("gmm-global-acc-stats --binary=false --update-flags=" + 
                                update_flags_str + " " + 
                                dubm + " " + 
                                shlex.quote(feats) + " " + 
                                tmp_spk_acc_file)
            args = shlex.split(acc_stats_command)
            p = subprocess.Popen(args) if self.debug else subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            p.wait()

            # 
            output_model = self.model_dir + "/" + spk_id + "-identity.gmm"
            map_command = ("gmm-global-est-map --update-flags=" + 
                        update_flags_str + " " + 
                        dubm + " " + 
                        tmp_spk_acc_file + " " + 
                        output_model)
            args = shlex.split(map_command)
            p = subprocess.Popen(args) if self.debug else subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            p.wait()

            # delete all the tmp file
            os.remove(tmp_spk_feats_scp)
            os.remove(tmp_spk_vad_scp)
            os.remove(tmp_spk_acc_file)

        print("--- obtaining gmm identity by updating ubm via MAP done ---")

        gmm_helper = gmm_ubm_kaldiHelper(pre_model_dir=self.gmm_model_dir, audio_dir=audio_dir_gmm, 
                                        mfcc_dir=mfcc_dir_gmm, log_dir=log_dir_gmm, score_dir=score_dir)

        model_path_list = []
        for spk_id in self.enroll_spk_id:
            model_path = self.model_dir + "/" + spk_id + "-identity.gmm"
            model_path_list.append(model_path)

        print("--- calculate z-norm mean, z-norm std and set threshold ---")

        ''' in order to set the threshold for gmm-ubm systems, we need the ubm model
        ''' 
        model_path_list_ubm = ([dubm] + model_path_list)
        score_array = gmm_helper.score_existing(model_path_list_ubm, self.illegal_utt_path, n_jobs=self.n_jobs, debug=self.debug)
        ubm_normed_score_array = score_array[:, 1:] - score_array[:, 0:1]
        score_untarget = ubm_normed_score_array.flatten()

        # clear directory, otherwise kaldi may not keep all the audios to be scored.
        if os.path.exists(audio_dir_gmm):
            shutil.rmtree(audio_dir_gmm)
        if os.path.exists(mfcc_dir_gmm):
            shutil.rmtree(mfcc_dir_gmm)
        if os.path.exists(log_dir_gmm):
            shutil.rmtree(log_dir_gmm)
        if os.path.exists(score_dir):
            shutil.rmtree(score_dir)

        if not os.path.exists(audio_dir_gmm):
            os.makedirs(audio_dir_gmm)
        if not os.path.exists(mfcc_dir_gmm):
            os.makedirs(mfcc_dir_gmm)
        if not os.path.exists(log_dir_gmm):
            os.makedirs(log_dir_gmm)
        if not os.path.exists(score_dir):
            os.makedirs(score_dir)

        score_target = []
        for spk_id, model_path in zip(self.enroll_spk_id, model_path_list_ubm[1:]):

            model_path_list_tmp = [model_path_list_ubm[0], model_path]
            test_utt_path_spk = list(np.array(self.test_utt_path)[np.argwhere(np.array(self.test_spk_id) == spk_id).flatten()])

            score_array = gmm_helper.score_existing(model_path_list_tmp, test_utt_path_spk, n_jobs=self.n_jobs, debug=self.debug)
            ubm_normed_score_array = score_array[:, 1] - score_array[:, 0]
            score_target += list(ubm_normed_score_array.flatten())

            # clear directory
            if os.path.exists(audio_dir_gmm):
                shutil.rmtree(audio_dir_gmm)
            if os.path.exists(mfcc_dir_gmm):
                shutil.rmtree(mfcc_dir_gmm)
            if os.path.exists(log_dir_gmm):
                shutil.rmtree(log_dir_gmm)
            if os.path.exists(score_dir):
                shutil.rmtree(score_dir)

            if not os.path.exists(audio_dir_gmm):
                os.makedirs(audio_dir_gmm)
            if not os.path.exists(mfcc_dir_gmm):
                os.makedirs(mfcc_dir_gmm)
            if not os.path.exists(log_dir_gmm):
                os.makedirs(log_dir_gmm)
            if not os.path.exists(score_dir):
                os.makedirs(score_dir)

        final_threshold, final_frr, final_far = self.set_threshold(score_target, score_untarget)

        print("-- threshold:%f, far:%f, frr:%f --" %(final_threshold, final_far, final_frr))

        ''' calculate z-norm and z-std. Note that z-norm is only used in CSI. IN SV and OSI, we use UBM norm.
        '''
        score_array = gmm_helper.score_existing(model_path_list, self.z_norm_utt_path, n_jobs=self.n_jobs, debug=self.debug)
        z_norm_means = np.mean(score_array, axis=0).flatten()
        z_norm_stds = np.std(score_array, axis=0).flatten()

        print("--- calculate z-norm mean, z-norm std and set threshold done ---")

        print(" --- dump speaker unique model --- ")
        for i, spk_id in enumerate(self.enroll_spk_id):

            utt_id = self.enroll_utt_id[i]
            identity_location = os.path.abspath(self.model_dir + "/" + spk_id + "-identity.gmm")
            z_norm_mean = z_norm_means[i]
            z_norm_std = z_norm_stds[i]

            # spk_unique_model = [spk_id, utt_id, identity_location, z_norm_mean, z_norm_std]
            spk_unique_model = [spk_id, utt_id, identity_location, z_norm_mean, z_norm_std, final_threshold]

            print(spk_unique_model),

            with open(self.model_dir + "/" + spk_id + ".gmm", "wb") as writer:
                pickle.dump(spk_unique_model, writer, protocol=-1)
        print(" --- dump speaker unique model done --- ")

    def build_ivector_model(self):
        #
        audio_dir = os.path.abspath("tmpfs/build_spk_models/iv-audio")
        if os.path.exists(audio_dir):
            shutil.rmtree(audio_dir)
        os.makedirs(audio_dir)
        #
        mfcc_dir = os.path.abspath("tmpfs/build_spk_models/iv-mfcc")
        if os.path.exists(mfcc_dir):
            shutil.rmtree(mfcc_dir)
        os.makedirs(mfcc_dir)
        #
        log_dir = os.path.abspath("tmpfs/build_spk_models/iv-log")
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir)
        #
        ivector_dir = os.path.abspath("tmpfs/build_spk_models/iv-socre")
        if os.path.exists(ivector_dir):
            shutil.rmtree(ivector_dir)
        os.makedirs(ivector_dir)

        trials = ivector_dir + "/trials"
        scores_file = ivector_dir + "/scores"
        ivector_scp = ivector_dir + "/ivector.scp"
        
        # 
        iv_helper = ivector_PLDA_kaldiHelper(pre_model_dir=self.ivector_model_dir, audio_dir=audio_dir, 
                        mfcc_dir=mfcc_dir, log_dir=log_dir, ivector_dir=ivector_dir)

        print("--- extracting and scoring ---")
        test_utt_id_scoring = self.z_norm_utt_id + self.test_utt_id + self.illegal_utt_id
        iv_helper.score_existing(self.audio_path_list, self.enroll_utt_id, spk_id_list=self.spk_id_list, 
                                utt_id_list=self.utt_id_list, test_utt_id=test_utt_id_scoring, 
                                n_jobs=self.n_jobs, flag=1, debug=self.debug)
        print("--- extracting and scoring done---")

        print("--- resolve score and obtain z norm mean and std value, and setting threshold ---")
        scores_mat = np.loadtxt(scores_file, dtype=str)
        train_utt_id = scores_mat[:, 0]
        test_utt_id_scoring = scores_mat[:, 1]
        score = scores_mat[:, 2].astype(np.float64)
        print('train_utt_id:', train_utt_id)
        print('test_utt_id_scoring:', test_utt_id_scoring)

        train_spk_id = np.array([utt_id.split("-")[0] for utt_id in train_utt_id])
        test_spk_id_scoring = np.array([utt_id.split("-")[0] for utt_id in test_utt_id_scoring])

        z_norm_index = []
        for i, utt_id in enumerate(test_utt_id_scoring):
            if utt_id in self.z_norm_utt_id:
                z_norm_index.append(i)
        target_index = np.argwhere(train_spk_id == test_spk_id_scoring).flatten()
        untarget_index = np.setdiff1d(np.argwhere(train_spk_id != test_spk_id_scoring).flatten(), np.array(z_norm_index))

        z_norm_means = np.zeros(len(self.enroll_utt_id), dtype=np.float64)
        z_norm_stds = np.zeros(len(self.enroll_utt_id), dtype=np.float64)
        score_target = []
        score_untarget =[]

        for i, id in enumerate(self.enroll_spk_id):
            index = np.argwhere(train_spk_id[z_norm_index] == id).flatten()
            mean = np.mean((score[z_norm_index])[index])
            std = np.std((score[z_norm_index])[index])
            z_norm_means[i] = mean
            z_norm_stds[i] = std

            index = np.argwhere(train_spk_id[target_index] == id).flatten()
            score_target += list(((score[target_index])[index] - mean) / std)

            index = np.argwhere(train_spk_id[untarget_index] == id).flatten()
            score_untarget += list(((score[untarget_index])[index] - mean) / std)

        final_threshold, final_frr, final_far = self.set_threshold(score_target, score_untarget)
        print("-- threshold:%f, far:%f, frr:%f --" %(final_threshold, final_far, final_frr))
        print("--- resolve score and obtain z norm mean and std value, and setting threshold done ---")

        print("--- dump speaker unique model ---")
        for i, utt_id in enumerate(self.enroll_utt_id):

            spk_id = self.enroll_spk_id[i]
            z_norm_mean = z_norm_means[i]
            z_norm_std = z_norm_stds[i]

            ivectors_utt_location = np.loadtxt(ivector_scp, dtype=str)
            ivectors_utt = ivectors_utt_location[:, 0]
            ivectors_location = ivectors_utt_location[:, 1]
            identity_location = os.path.abspath(ivectors_location[np.argwhere(ivectors_utt == utt_id).flatten()[0]]) # use absolute path

            spk_unique_model = [spk_id, utt_id, identity_location, z_norm_mean, z_norm_std, final_threshold]
            print(spk_unique_model),

            with open(self.model_dir + "/" + spk_id + ".iv", "wb") as writer:
                pickle.dump(spk_unique_model, writer, protocol=-1)
        print("--- dump speaker unique model done ---")

    def build_xvector_model(self):
        # need to remove build_spk_models dir before building model
        xv_audio_dir = os.path.abspath("tmpfs/build_spk_models/xv-audio")
        if os.path.exists(xv_audio_dir):
            shutil.rmtree(xv_audio_dir)
        os.makedirs(xv_audio_dir)
        xv_mfcc_dir = os.path.abspath("tmpfs/build_spk_models/xv-mfcc")
        if os.path.exists(xv_mfcc_dir):
            shutil.rmtree(xv_mfcc_dir)
        os.makedirs(xv_mfcc_dir)
        xv_log_dir = os.path.abspath("tmpfs/build_spk_models/xv-log")
        if os.path.exists(xv_log_dir):
            shutil.rmtree(xv_log_dir)
        os.makedirs(xv_log_dir)
        xvector_dir = os.path.abspath("tmpfs/build_spk_models/xv-score")
        if os.path.exists(xvector_dir):
            shutil.rmtree(xvector_dir)
        os.makedirs(xvector_dir)

        xv_scores_file = xvector_dir + "/scores"
        xvector_scp = xvector_dir + "/xvector.scp"

        print("----- generate xvector identity and corresponding speaker model, setting threshold -----")
        xv_helper = xvector_PLDA_kaldiHelper(pre_model_dir=self.xvector_model_dir, audio_dir=xv_audio_dir, \
                        mfcc_dir=xv_mfcc_dir, log_dir=xv_log_dir, xvector_dir=xvector_dir)

        print("--- extracting and scoring ---")
        test_utt_id = self.z_norm_utt_id + self.test_utt_id + self.illegal_utt_id
        xv_helper.score_existing(self.audio_path_list, self.enroll_utt_id, spk_id_list=self.spk_id_list, 
                                utt_id_list=self.utt_id_list, test_utt_id=test_utt_id, 
                                n_jobs=self.n_jobs, flag=1, debug=self.debug)
        print("--- extracting and scoring done---")

        print("--- resolve score and obtain z norm mean and std value, and setting threshold ---")
        scores_mat = np.loadtxt(xv_scores_file, dtype=str)
        train_utt_id = scores_mat[:, 0]
        test_utt_id = scores_mat[:, 1]
        print('train_utt_id:',train_utt_id)
        print('test_utt_id:',test_utt_id)
        score = scores_mat[:, 2].astype(np.float64)
        train_spk_id = np.array([utt_id.split("-")[0] for utt_id in train_utt_id])
        test_spk_id = np.array([utt_id.split("-")[0] for utt_id in test_utt_id])
        print('train_spk_id:',train_spk_id)
        print('test_spk_id:',test_spk_id)

        z_norm_index = []
        for i, utt_id in enumerate(test_utt_id):
            if utt_id in self.z_norm_utt_id:
                z_norm_index.append(i)
        target_index = np.argwhere(train_spk_id == test_spk_id).flatten()
        unequal_index = np.argwhere(train_spk_id != test_spk_id).flatten()
        untarget_index = np.setdiff1d(unequal_index, np.array(z_norm_index))
        print('target_index:',target_index)
        print('unequal_index:',unequal_index)
        print('z_norm_index:',z_norm_index)
        print('untarget_index:',untarget_index)

        z_norm_means = np.zeros(len(self.enroll_utt_id), dtype=np.float64)
        z_norm_stds = np.zeros(len(self.enroll_utt_id), dtype=np.float64)
        score_target = []
        score_untarget = []

        for i, id in enumerate(self.enroll_spk_id):
            # 使用enroll_spk与z_norm中utt的比对score，计算mean和std
            index = np.argwhere(train_spk_id[z_norm_index] == id).flatten()
            mean = np.mean((score[z_norm_index])[index])
            std = np.std((score[z_norm_index])[index])
            z_norm_means[i] = mean
            z_norm_stds[i] = std

            # 使用target_index中，当前id的score，做标准化以后，将scores添加到score_target
            index = np.argwhere(train_spk_id[target_index] == id).flatten()
            score_target += list(((score[target_index])[index] - mean) / std)

            # 使用untarget_index中，当前id的score，做标准化以后，将scores添加到score_untarget
            index = np.argwhere(train_spk_id[untarget_index] == id).flatten()
            score_untarget += list(((score[untarget_index])[index] - mean) / std)

        # 使用score_target和score_untarget计算阈值
        final_threshold, final_frr, final_far = self.set_threshold(score_target, score_untarget)
        print("-- threshold:%f, far:%f, frr:%f --" %(final_threshold, final_far, final_frr))
        # -- threshold:2.092117, far:1.414141, frr:1.414141 --
        print("--- resolve score and obtain z norm mean and std value, and setting threshold done ---")
        print("--- dump speaker unique model ---")

        for i, utt_id in enumerate(self.enroll_utt_id):
            spk_id = self.enroll_spk_id[i]
            z_norm_mean = z_norm_means[i]
            z_norm_std = z_norm_stds[i]

            xvectors_utt_location = np.loadtxt(xvector_scp, dtype=str)
            xvectors_utt = xvectors_utt_location[:, 0]
            xvectors_location = xvectors_utt_location[:, 1]
            identity_location = os.path.abspath(xvectors_location[np.argwhere(xvectors_utt == utt_id).flatten()[0]]) # use absolute path

            spk_unique_model = [spk_id, utt_id, identity_location, z_norm_mean, z_norm_std, final_threshold]
            print(spk_unique_model),
            '''
            ['2830', '2830-3980-0075', '/data1/github/fakebob/tmpfs/build_spk_models/xv-xvector/xvector.12.ark:182439', -20.121085328762472, 11.247744770031366, 2.092116990976621]
            '''
            with open(self.model_dir + "/" + spk_id + ".xv", "wb") as writer:
                pickle.dump(spk_unique_model, writer, protocol=-1)
        print("--- dump speaker unique model done ---")



#----------------------------------------------------------------------------

_examples = '''examples:

  # build speaker model
  python %(prog)s --enroll=./data/tq-enrollment/ --illegal=./data/tq-illegal --test=./data/tq-test --znorm=./data/tq-z-norm --type=xvector --num=33

  #
  python %(prog)s --enroll=./data/0-enrollment/ --illegal=./data/0-illegal --test=./data/0-test --znorm=./data/0-z-norm --type=xvector --num=43

'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''Speaker model generation.

Run 'python %(prog)s --help' for help info.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--enroll', type=str, default='./data/enrollment-set', help='enrollment directory')
    parser.add_argument('--illegal', type=str, default='./data/illegal-set', help='illegal directory')
    parser.add_argument('--test', type=str, default='./data/test-set', help='test directory')
    parser.add_argument('--znorm', type=str, default='./data/z-norm-set', help='znorm directory')
    parser.add_argument('--type', type=str, default='xvector', help='supported type, ivector|xvector|gmm, default is xvector')
    parser.add_argument('--num', type=int, default=48, help='number of job, default 48')
    args = parser.parse_args()
    
    builder = SpeakerModelBuilder(args.enroll, args.illegal, args.test, args.znorm, args.num)
    builder.load_data()
    if args.type == 'gmm':
        builder.build_gmm_model()
    elif args.type == 'ivector':
        builder.build_ivector_model()
    elif args.type == 'xvector':
        builder.build_xvector_model()
    else:
        print('Unsupported model type')


if __name__ == '__main__':
    main()

