import os
import pickle
import numpy as np
from scipy.io.wavfile import read, write

from gmm_ubm_CSI import gmm_CSI
from gmm_ubm_OSI import gmm_OSI
from gmm_ubm_SV import gmm_SV
from ivector_PLDA_CSI import iv_CSI
from ivector_PLDA_OSI import iv_OSI
from ivector_PLDA_SV import iv_SV
from xvector_PLDA_SV import xv_SV

from defines import *

def load_model(spk_id_list, architecture, task, id):
    '''
    '''
    id = os.path.join('tmpfs', id)

    if architecture == IV:
        iv_model_paths = [os.path.join(model_dir, spk_id + ".iv") for spk_id in spk_id_list]
        iv_model_list = []
        for path in iv_model_paths:
            with open(path, "rb") as reader:
                model = pickle.load(reader)
                iv_model_list.append(model)

        if task == OSI:
            model = iv_OSI(id, iv_model_list)
        elif task == CSI:
            model = iv_CSI(id, iv_model_list)
        else:
            model = iv_SV(id, iv_model_list[0])
    elif architecture == XV:
        xv_model_paths = [os.path.join(model_dir, spk_id + ".xv") for spk_id in spk_id_list]
        xv_model_list = []
        for path in xv_model_paths:
            with open(path, "rb") as reader:
                model = pickle.load(reader)
                xv_model_list.append(model)

        if task == OSI:
            print('Unimplemented Model!')
            exit(1)
            # model = xv_OSI(id, xv_model_list)
        elif task == CSI:
            print('Unimplemented Model!')
            exit(1)
            # model = xv_CSI(id, xv_model_list)
        else:
            model = xv_SV(id, xv_model_list[0])
    else:
        gmm_model_paths = [os.path.join(model_dir, spk_id + ".gmm") for spk_id in spk_id_list]
        gmm_model_list = []
        for path in gmm_model_paths:
            with open(path, "rb") as reader:
                model = pickle.load(reader)
                gmm_model_list.append(model)

        ubm = os.path.join("kaldi_models", "gmm_models", "final.dubm")
        if task == OSI:
            model = gmm_OSI(id, gmm_model_list, ubm)
        elif task == CSI:
            model = gmm_CSI(id, gmm_model_list)
        else:
            model = gmm_SV(id, gmm_model_list[0], ubm)

    return model


def load_data(adver_audio_dir, checkpoint_dir, task, attack_type, model, spk_id_list,
                illegal_dir, test_dir, n_jobs=10, debug=False):
    audio_names = []
    adver_audio_paths = []
    checkpoint_paths = []

    if task == CSI:
        spk_ids = np.array(model.spk_ids)
        data_path = test_dir
        audio_list = []
        true_label_list = []
        spk_iter = os.listdir(data_path)
        for spk_id in spk_iter:
            true_label = np.argwhere(spk_ids == spk_id).flatten()[0]
            spk_dir = os.path.join(data_path, spk_id)
            audio_iter = os.listdir(spk_dir)
            adver_audio_spk_dir = os.path.join(adver_audio_dir, spk_id)
            if not os.path.exists(adver_audio_spk_dir):
                os.makedirs(adver_audio_spk_dir)
            checkpoint_spk_dir = os.path.join(checkpoint_dir, spk_id)
            if not os.path.exists(checkpoint_spk_dir):
                os.makedirs(checkpoint_spk_dir)
            for audio_name in audio_iter:
                audio_names.append(audio_name)
                adver_audio_paths.append(os.path.join(os.path.join(adver_audio_dir, spk_id), audio_name))
                checkpoint_paths.append(os.path.join(os.path.join(checkpoint_dir, spk_id), audio_name.split(".")[0] + ".cp"))
                audio_path = os.path.join(spk_dir, audio_name)
                _, audio = read(audio_path)
                audio = audio / (2 ** (bits_per_sample - 1))
                audio_list.append(audio)
                true_label_list.append(true_label)
        
        # skip those wrongly classified
        decisions, _ = model.make_decisions(audio_list, fs=fs, bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
        preserve_index = np.argwhere(np.array(decisions) == true_label_list).flatten()
        audio_list = [audio_list[index] for index in preserve_index]
        true_label_list = [true_label_list[index] for index in preserve_index]
        audio_names = [audio_names[index] for index in preserve_index]
        adver_audio_paths = [adver_audio_paths[index] for index in preserve_index]
        checkpoint_paths = [checkpoint_paths[index] for index in preserve_index]

        if attack_type == UNTARGETED:
            return audio_list, true_label_list, None, audio_names, adver_audio_paths, checkpoint_paths
        
        audio_list_targeted = []
        audio_names_targeted = []
        adver_audio_paths_targeted = []
        checkpoint_paths_targeted = []
        true_label_list_targeted = []
        target_label_list = []
        for audio, true_label, audio_name, adver_audio_path, checkpoint_path in zip(audio_list, 
                                                                                    true_label_list, 
                                                                                    audio_names, 
                                                                                    adver_audio_paths, 
                                                                                    checkpoint_paths):

            for target_label in range(len(model.spk_ids)):
                if target_label == true_label:
                    continue
                audio_list_targeted.append(audio)
                audio_names_targeted.append(audio_name)
                adver_audio_paths_targeted.append(adver_audio_path.split(".")[0] + "_" + str(target_label) + ".wav")
                checkpoint_paths_targeted.append(checkpoint_path.split(".")[0] + "_" + str(target_label) + ".cp")
                true_label_list_targeted.append(true_label)
                target_label_list.append(target_label)
        
        audio_names = audio_names_targeted
        adver_audio_paths = adver_audio_paths_targeted
        checkpoint_paths = checkpoint_paths_targeted
        return audio_list_targeted, true_label_list_targeted, target_label_list, audio_names, adver_audio_paths, checkpoint_paths

    elif task == OSI:
        data_path = illegal_dir
        audio_list = []
        spk_iter = os.listdir(data_path)
        for spk_id in spk_iter:
            spk_dir = os.path.join(data_path, spk_id)
            audio_iter = os.listdir(spk_dir)
            adver_audio_spk_dir = os.path.join(adver_audio_dir, spk_id)
            if not os.path.exists(adver_audio_spk_dir):
                os.makedirs(adver_audio_spk_dir)
            checkpoint_spk_dir = os.path.join(checkpoint_dir, spk_id)
            if not os.path.exists(checkpoint_spk_dir):
                os.makedirs(checkpoint_spk_dir)
            for audio_name in audio_iter:
                audio_names.append(audio_name)
                adver_audio_paths.append(os.path.join(os.path.join(adver_audio_dir, spk_id), audio_name))
                checkpoint_paths.append(os.path.join(os.path.join(checkpoint_dir, spk_id), audio_name.split(".")[0] + ".cp"))
                audio_path = os.path.join(spk_dir, audio_name)
                _, audio = read(audio_path)
                audio = audio / (2 ** (bits_per_sample - 1))
                audio_list.append(audio)
        
        # skip those far audios
        decisions, _ = model.make_decisions(audio_list, fs=fs, bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
        preserve_index = np.argwhere(np.array(decisions) == -1).flatten()
        audio_list = [audio_list[index] for index in preserve_index]
        audio_names = [audio_names[index] for index in preserve_index]
        adver_audio_paths = [adver_audio_paths[index] for index in preserve_index]
        checkpoint_paths = [checkpoint_paths[index] for index in preserve_index]

        if attack_type == UNTARGETED:
            return audio_list, None, None, audio_names, adver_audio_paths, checkpoint_paths
        
        audio_list_targeted = []
        audio_names_targeted = []
        adver_audio_paths_targeted = []
        checkpoint_paths_targeted = []
        target_label_list = []
        for audio, audio_name, adver_audio_path, checkpoint_path in zip(audio_list,
                                                                        audio_names,
                                                                        adver_audio_paths,
                                                                        checkpoint_paths):

            for target_label in range(len(model.spk_ids)):
                audio_list_targeted.append(audio)
                audio_names_targeted.append(audio_name)
                adver_audio_paths_targeted.append(adver_audio_path.split(".")[0] + "_" + str(target_label) + ".wav")
                checkpoint_paths_targeted.append(checkpoint_path.split(".")[0] + "_" + str(target_label) + ".cp")
                target_label_list.append(target_label)
        
        audio_names = audio_names_targeted
        adver_audio_paths = adver_audio_paths_targeted
        checkpoint_paths = checkpoint_paths_targeted
        return audio_list_targeted, None, target_label_list, audio_names, adver_audio_paths, checkpoint_paths
    
    else: # SV
        audio_list = []
        data_path = illegal_dir
        spk_iter = os.listdir(data_path)
        for spk_id in spk_iter:
            spk_dir = os.path.join(data_path, spk_id)
            audio_iter = os.listdir(spk_dir)
            adver_audio_spk_dir = os.path.join(adver_audio_dir, spk_id)
            if not os.path.exists(adver_audio_spk_dir):
                os.makedirs(adver_audio_spk_dir)
            checkpoint_spk_dir = os.path.join(checkpoint_dir, spk_id)
            if not os.path.exists(checkpoint_spk_dir):
                os.makedirs(checkpoint_spk_dir)
            for audio_name in audio_iter:
                audio_names.append(audio_name)
                adver_audio_paths.append(os.path.join(os.path.join(adver_audio_dir, spk_id), audio_name))
                checkpoint_paths.append(os.path.join(os.path.join(checkpoint_dir, spk_id), audio_name.split(".")[0] + ".cp"))
                audio_path = os.path.join(spk_dir, audio_name)
                print('audio_path:',audio_path)
                _, audio = read(audio_path)
                audio = audio / (2 ** (bits_per_sample - 1)) # convert to [-1,1)
                audio_list.append(audio)
        
        # skip those far audios
        decisions, _ = model.make_decisions(audio_list, fs=fs, bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
        print('decisions:', decisions)
        # print(np.array(decisions))
        preserve_index = np.argwhere(np.array(decisions) == -1).flatten()
        # print('preserve_index:', preserve_index)
        audio_list = [audio_list[index] for index in preserve_index]
        audio_names = [audio_names[index] for index in preserve_index]
        adver_audio_paths = [adver_audio_paths[index] for index in preserve_index]
        checkpoint_paths = [checkpoint_paths[index] for index in preserve_index]
        return audio_list, None, None, audio_names, adver_audio_paths, checkpoint_paths
