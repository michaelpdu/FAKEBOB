import os
import argparse

class TianqiongDataGenerator:
    '''
    '''
    def __init__(self, tq_dir):
        self.tq_dir = tq_dir
        self.tq_utt_id = []
        self.tq_spk_id = []
        self.tq_utt_path = []

    def load_data(self):
        audio_iter = os.listdir(self.tq_dir)
        for i, audio_name in enumerate(audio_iter):
            utt_id = audio_name.split(".")[0]
            spk_id = utt_id.split("-")[0]
            path = os.path.join(self.tq_dir, audio_name)
            self.tq_utt_path.append(path)
            self.tq_utt_id.append(utt_id)
            self.tq_spk_id.append(spk_id)
        print(self.tq_utt_id)
        print(self.tq_spk_id)

    def generate(self):
        pass

#----------------------------------------------------------------------------

_examples = '''examples:

  # generate Tianqiong voice data
  python %(prog)s

'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''Generate Tianqiong Voice Data.

Run 'python %(prog)s --help' for help info.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--tq_dir', type=str, default='./data/tq', help='tq voice directory')
    args = parser.parse_args()
    
    data_gen = TianqiongDataGenerator(args.tq_dir)
    data_gen.load_data()

if __name__ == '__main__':
    main()
