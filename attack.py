
import argparse
import os
import pickle
import random
import numpy as np
from scipy.io.wavfile import read, write
from FAKEBOB import FakeBob
from utility import load_model, load_data
from defines import *

# def attack(spk_id_list, architecture, task, attack_type, adver_thresh,
#          epsilon, max_iter, max_lr, min_lr, samples_per_draw, sigma,
#          momentum, plateau_length, plateau_drop, n_jobs, debug, start, end, step):
def attack(spk_id_list, architecture, task, attack_type, adver_thresh, threshold_delta,
         epsilon, max_iter, max_lr, min_lr, samples_per_draw, sigma,
         momentum, plateau_length, plateau_drop, 
         illegal_dir, test_dir,
         n_jobs, debug):
    
    id = "{}-{}-{}".format(architecture, task, attack_type)
    adver_audio_dir = os.path.join('tmpfs', "adversarial-audio", id)
    checkpoint_dir = os.path.join('tmpfs', "checkpoint", id)
    # new id for attack only
    id = id+"-attack"

    if task == SV:
        adver_audio_dir = os.path.join(adver_audio_dir, spk_id_list[0])
        checkpoint_dir = os.path.join(checkpoint_dir, spk_id_list[0])

    if not os.path.exists(adver_audio_dir):
        os.makedirs(adver_audio_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # global audio_names
    # audio_names = []
    # global adver_audio_paths
    # adver_audio_paths = []
    # global checkpoint_paths
    # checkpoint_paths = []

    print('------ load model ------')
    model = load_model(spk_id_list, architecture, task, id)
    print('model threshold:', model.threshold)

    print('------ load data ------')
    audio_list, true_label_list, target_label_list, \
    audio_names, adver_audio_paths, checkpoint_paths = load_data(
        adver_audio_dir, checkpoint_dir, task, attack_type, model, spk_id_list,
        illegal_dir, test_dir, n_jobs=n_jobs, debug=debug)
    total_cnt = len(audio_list)
    
    print('------ total audio count: %d ------' % total_cnt)
    
    success_cnt = 0
    # print("----- fakebob initialize -----")
    fake_bob = FakeBob(task, attack_type, model, adver_thresh=adver_thresh, epsilon=epsilon, max_iter=max_iter,
                max_lr=max_lr, min_lr=min_lr, samples_per_draw=samples_per_draw, sigma=sigma, momentum=momentum, 
                plateau_length=plateau_length, plateau_drop=plateau_drop)

    if task == CSI:
        if attack_type == TARGETED:
            for audio, true_label, \
                target_label, audio_name, \
                adver_audio_path, checkpoint_path in zip(audio_list, true_label_list, 
                                                        target_label_list, audio_names, 
                                                        adver_audio_paths, checkpoint_paths):
                print("--- %s, %s, %s, audio name:%s, true spk:%s, target spk:%s ---" %(architecture, task, attack_type, audio_name, model.spk_ids[true_label], model.spk_ids[target_label]))
                adver_audio, success_flag = fake_bob.attack(audio, checkpoint_path, target=target_label, fs=fs, 
                                                            bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
                write(adver_audio_path, fs, adver_audio)
                if success_flag == 1:
                    success_cnt += 1
        else:
            for audio, true_label, audio_name, \
                adver_audio_path, checkpoint_path in zip(audio_list, true_label_list, audio_names, 
                                                        adver_audio_paths, checkpoint_paths):
                print("--- %s, %s, %s, audio name:%s, true spk:%s ---" %(architecture, task, attack_type, audio_name, model.spk_ids[true_label]))
                adver_audio, success_flag = fake_bob.attack(audio, checkpoint_path, true=true_label, fs=fs, 
                                                            bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
                write(adver_audio_path, fs, adver_audio)
                if success_flag == 1:
                    success_cnt += 1
    elif task == OSI:
        # first estimates the threshold
        audio = audio_list[np.random.choice(total_cnt, 1)[0]] # randomly choose an audio to estimate the threshold
        threshold_estimated, _, _ = fake_bob.estimate_threshold(audio, fs=fs, bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
        print('threshold_estimated:', threshold_estimated)
        # threshold_estimated = 1.6
        if attack_type == TARGETED:
            for audio, target_label, audio_name, \
                adver_audio_path, checkpoint_path in zip(audio_list, 
                                                        target_label_list, audio_names, 
                                                        adver_audio_paths, checkpoint_paths):
                
                print("--- %s, %s, %s, audio name:%s, target spk:%s ---" %(architecture, task, attack_type, audio_name, model.spk_ids[target_label]))
                adver_audio, success_flag = fake_bob.attack(audio, checkpoint_path, threshold=threshold_estimated, target=target_label, fs=fs, 
                                                            bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
                
                write(adver_audio_path, fs, adver_audio)
                if success_flag == 1:
                    success_cnt += 1
        else:
            for audio, audio_name, \
                adver_audio_path, checkpoint_path in zip(audio_list, audio_names, 
                                                        adver_audio_paths, checkpoint_paths):
                print("--- %s, %s, %s, audio name:%s ---" %(architecture, task, attack_type, audio_name))
                adver_audio, success_flag = fake_bob.attack(audio, checkpoint_path, threshold=threshold_estimated, fs=fs, 
                                                            bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
                write(adver_audio_path, fs, adver_audio)
                if success_flag == 1:
                    success_cnt += 1
    else:
        print('>>>>>>>>>>>>> estimate threshold >>>>>>>>>>>>>')
        audio = audio_list[np.random.choice(total_cnt, 1)[0]] # randomly choose an audio to estimate the threshold
        #threshold_estimated, _, _ = fake_bob.estimate_threshold(audio, fs=fs, bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
        #threshold_estimated = threshold_estimated + threshold_delta
        threshold_estimated = model.threshold + 0.001 + threshold_delta # 9001-0000-0001
        # threshold_estimated = 0.5 + threshold_delta
        print('threshold_estimated:', threshold_estimated, '\n')
        # threshold_estimated = 1.6

        print('>>>>>>>>>>>>> generate adv. examples >>>>>>>>>>>>>')
        for audio, audio_name, \
            adver_audio_path, checkpoint_path in zip(audio_list, audio_names, 
                                                    adver_audio_paths, checkpoint_paths):
            print("\n--- generate adv. example [%s, %s, %s, audio name:%s] ---" %(architecture, task, attack_type, audio_name))

            adver_audio, success_flag = fake_bob.attack(audio, checkpoint_path, threshold=threshold_estimated, fs=fs, 
                                                        bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
            path_wo_ext, ext = os.path.splitext(adver_audio_path)
            adver_audio_path = "{}-advthresh{}{}".format(path_wo_ext, adver_thresh, ext)
            print('[*] write audio to:', adver_audio_path)
            write(adver_audio_path, fs, adver_audio)
            if success_flag == 1:
                success_cnt += 1

    print('------ attack successful rate %d ------' %(success_cnt * 100 / total_cnt))
    print("----- generate adversarial voices done -----")

_examples = '''examples:

  # attack 
  python %(prog)s --speaker_id=9001

'''

if __name__ == "__main__":
    # 
    parser = argparse.ArgumentParser(
        description='''attack process.

Run 'python %(prog)s --help' for help info.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    #
    parser.add_argument("--speaker_id", "-spk_id", nargs="+", type=str, help="speaker_id list, seprated by ',', and first one will be used when task is SV.")
    #
    parser.add_argument("--architecture", "-archi", default=XV, choices=[GMM, IV, XV], type=str, help="support gmm|iv|xv, and default is XV")
    parser.add_argument("--task", "-task", default=SV, choices=[OSI, CSI, SV], type=str, help="support OSI|CSI|SV, and default is SV")
    parser.add_argument("--attack_type", "-type", default=TARGETED, choices=[UNTARGETED, TARGETED], type=str, 
                        help="support TARGETED|UNTARGETED, and default is TARGETED. Omit if task is SV") # obmit when task is SV
    #
    parser.add_argument("--adver_thresh", "-adver", default=0., type=float, help="default is 0.")
    parser.add_argument("--epsilon", "-epsilon", default=0.002, type=float, help="default is 0.002")
    parser.add_argument("--max_iter", "-max_iter", default=1000, type=int, help="default is 1000")
    parser.add_argument("--max_lr", "-max_lr", default=0.0005, type=float, help="default is 0.0005")
    parser.add_argument("--min_lr", "-min_lr", default=1e-6, type=float, help="default is 1e-6")
    parser.add_argument("--samples_per_draw", "-samples", default=46, type=int, 
                        help="how many noised samples would be generated, when calcute gradient. default is 46")
    parser.add_argument("--sigma", "-sigma", default=0.001, type=float, help="default is 0.001")
    parser.add_argument("--momentum", "-momentum", default=0.9, type=float, help="default is 0.9")
    parser.add_argument("--plateau_length", "-plateau_length", default=5, type=int, help="default is 5")
    parser.add_argument("--plateau_drop", "-plateau_drop", default=2.0, type=float, help="default is 2.0")
    #
    parser.add_argument("--n_jobs", "-nj", default=48, type=int, help="default is 48")
    parser.add_argument("--debug", "-debug", default=False, type=bool, help="default is False")

    # parser.add_argument("--start", "-start", default=0, type=int)
    # parser.add_argument("--end", "-end", default=-1, type=int)
    # parser.add_argument("--step", "-step", default=1, type=int)
    #
    # parser.add_argument('--enroll', type=str, default='./data/tq-enrollment', help='enrollment directory')
    parser.add_argument('--illegal', type=str, default='./data/tq-illegal', help='illegal directory')
    parser.add_argument('--test', type=str, default='./data/tq-test', help='test directory')
    # parser.add_argument('--znorm', type=str, default='./data/tq-z-norm', help='znorm directory')

    args = parser.parse_args()

    spk_id_list = args.speaker_id
    architecture = args.architecture
    task = args.task
    attack_type = args.attack_type
    if task == SV:
        attack_type = TARGETED
        spk_id_list = spk_id_list[0:1] # SV only support one enrolled speakers
    
    adver_thresh = args.adver_thresh
    epsilon = args.epsilon
    max_iter = args.max_iter
    max_lr = args.max_lr
    min_lr = args.min_lr
    samples_per_draw = args.samples_per_draw
    sigma = args.sigma
    momentum = args.momentum
    plateau_length = args.plateau_length
    plateau_drop = args.plateau_drop
    threshold_delta = 0.0

    n_jobs = args.n_jobs
    debug = args.debug

    # start = args.start
    # end = args.stop
    # step = args.step

    # attack(spk_id_list, architecture, task, attack_type, adver_thresh,
    #      epsilon, max_iter, max_lr, min_lr, samples_per_draw, sigma,
    #      momentum, plateau_length, plateau_drop, n_jobs, debug, start, end, step)
    attack(spk_id_list, architecture, task, attack_type, adver_thresh, threshold_delta,
         epsilon, max_iter, max_lr, min_lr, samples_per_draw, sigma,
         momentum, plateau_length, plateau_drop, 
         args.illegal, args.test,
         n_jobs, debug)
