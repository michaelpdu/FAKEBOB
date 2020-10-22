from utility import load_model
from scipy.io.wavfile import read
from defines import *

import os
import argparse

class UttVerifier:
    """
    """

    def __init__(self, arch, target_spk_id, debug=True):
        print('> arch:', arch, ', spk_id:', target_spk_id, ', debug:', debug)
        self.debug = debug
        spk_id_list = [target_spk_id]
        architecture = arch
        task = "SV"
        attack_type = "targeted"
        id = "{}-{}-{}-{}".format(architecture, task, attack_type, "uttverifer")
        self.model = load_model(spk_id_list, architecture, task, id)
        print('threshold:', self.model.threshold)

    def verify_utt(self, utt_path):
        audio_list = []
        _, audio = read(utt_path)
        audio = audio / (2 ** (bits_per_sample - 1)) # normalization from 32768 to 1.
        audio_list.append(audio)
        decisions, scores = self.model.make_decisions(audio_list, fs=fs, bits_per_sample=bits_per_sample, n_jobs=1, debug=self.debug)
        print('decisions:', decisions, ', scores:', scores)

    def verify_dir(self, dir_path):
        utt_path_list = []
        audio_list = []
        for root, dirs, files in os.walk(dir_path):
            for name in files:
                name_wo_ext, ext = os.path.splitext(name)
                if ext.lower() == '.wav':
                    utt_path = os.path.join(root, name)
                    try:
                        utt_path_list.append(utt_path)
                        _, audio = read(utt_path)
                        audio = audio / (2 ** (bits_per_sample - 1))
                        audio_list.append(audio)
                    except Exception as e:
                        print(e)
                        print(utt_path)
        count = len(audio_list)
        print('There are {} utterance files in {}'.format(count, dir_path))
        n_jobs = count if count <= 48 else 48
        decisions, scores = self.model.make_decisions(audio_list, fs=fs, bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=self.debug)
        for i in range(count):
            print('{}, {}, {}'.format(utt_path_list[i], decisions[i], scores[i]))

    def verify(self, target):
        if os.path.isdir(target):
            self.verify_dir(target)
        elif os.path.isfile(target):
            self.verify_utt(target)
        else:
            pass

_examples = """
  
  # verify adv. example 260-123286-0006.wav, generated on xvector-PLDA, on gmm-ubm model
  python %(prog)s 'gmm' 1580 './tmpfs/adversarial-audio/xv-SV-targeted/1580/260/260-123286-0006.wav'

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Verify Transferability',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('arch', type=str, help='support iv, xv, gmm and all')
    parser.add_argument('spk_id', type=str, default='1580', help='target speaker id')
    parser.add_argument('audio_path', type=str, help='adversarial audio path')
    args = parser.parse_args()
    
    if args.arch == 'all':
        # verify('gmm', args.spk_id, args.audio_path)
        # verify('iv', args.spk_id, args.audio_path)
        # verify('xv', args.spk_id, args.audio_path)
        pass
    else:
        helper = UttVerifier(args.arch, args.spk_id)
        helper.verify(args.audio_path)
