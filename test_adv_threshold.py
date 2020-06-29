import os
from attackMain import attack
from verify_transferability import verify
from defines import *

spk_ids=["1580"]

#archi=gmm
#archi=iv
archi=XV
#task=OSI
#task=CSI
task=SV
#attack_type=targeted
attack_type=UNTARGETED

id = '{}-{}-{}'.format(archi,task,attack_type)

adver_thresh=0.0
threshold_delta=0.0
epsilon=0.002
max_iter=1000
max_lr=0.0005
min_lr=1e-6
samples=50
sigma=0.001
momentum=0.9
plateau_length=5
plateau_drop=2.0

n_jobs=48
debug=False

print('modify adver_thresh...')
for i in range(1,10):
    print('>>>>>>>>>>>>> i = {} >>>>>>>>>>>>>'.format(i))
    delta = 0.1*i
    attack(spk_ids, archi, task, attack_type, adver_thresh+delta, threshold_delta,
        epsilon, max_iter, max_lr, min_lr, samples, sigma,
        momentum, plateau_length, plateau_drop, n_jobs, debug)
    
    ori_adv = './tmpfs/adversarial-audio/{}/1580/260/260-123286-0006.wav'.format(id)
    new_adv = './tmpfs/adversarial-audio/{}/1580/260/260-123286-0006-t1-{}.wav'.format(id,i)
    os.rename(ori_adv, new_adv)

    print('>> verify transferability >>')
    verify('gmm', '1580', new_adv)
    verify('iv', '1580', new_adv)
    verify('xv', '1580', new_adv)

print('modify threshold_delta...')
for i in range(1,10):
    print('>>>>>>>>>>>>> i = {} >>>>>>>>>>>>>'.format(i))
    delta = 0.1*i
    attack(spk_ids, archi, task, attack_type, adver_thresh, threshold_delta+delta,
        epsilon, max_iter, max_lr, min_lr, samples, sigma,
        momentum, plateau_length, plateau_drop, n_jobs, debug)
    
    ori_adv = './tmpfs/adversarial-audio/{}/1580/260/260-123286-0006.wav'.format(id)
    new_adv = './tmpfs/adversarial-audio/{}/1580/260/260-123286-0006-t2-{}.wav'.format(id,i)
    os.rename(ori_adv, new_adv)

    print('>> verify transferability >>')
    verify('gmm', '1580', new_adv)
    verify('iv', '1580', new_adv)
    verify('xv', '1580', new_adv)