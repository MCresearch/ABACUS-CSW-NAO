'''
after downloading and before archiving, it is recommended to delete all orbital
and psp files to save space.
'''
import logging.config
import os
import shutil
import logging
import time


def compress(folder, flavor = 'tar.gz'):
    '''compress the folder'''
    import tarfile
    import zipfile
    if flavor not in ['tar.gz', 'tar', 'zip']:
        raise ValueError('flavor not supported')
    
    fn = f'{folder}.{flavor}'
    if flavor == 'tar.gz':
        with tarfile.open(fn, 'w:gz') as tar:
            tar.add(folder)
    elif flavor == 'tar':
        with tarfile.open(fn, 'w') as tar:
            tar.add(folder)
    elif flavor == 'zip':
        with zipfile.ZipFile(fn, 'w') as z:
            for root, _, files in os.walk(folder):
                for f in files:
                    z.write(os.path.join(root, f))
    return fn

def _read_abacus_input(fn):
    '''read the abacus input in a very rude way'''
    def comp(l):
        w = l.split()
        return (w[0], ' '.join(w[1:]))
    
    with open(fn, 'r') as f:
        lines = [line.strip() for line in f.readlines() \
                 if not line.startswith('INPUT_PARAMETERS')\
                    and not line.startswith('#')]
    lines = [l.split('#')[0] for l in lines if l]
    lines = [comp(l) for l in lines]
    return dict(lines)

def _get_from_jobdir(folder):
    '''get the name of pp_orb folder from jobdir'''
    params = _read_abacus_input(os.path.join(folder, 'INPUT'))
    return params.get('pseudo_dir') or params.get('orbital_dir')

def find_pporb_folder(root):
    '''there are two possibilities, the first is directly get the
    pporb dir, the second is to get the pporb dir from jobdir'''
    first = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))][0]
    found = any([f.endswith('.orb') for f in os.listdir(os.path.join(root, first))])
    if found:
        return os.path.basename(first)
    else:
        temp = _get_from_jobdir(os.path.join(root, first))
        return os.path.basename(temp)

def main(target):
    '''clean the pporb folders'''
    logging.basicConfig(level=logging.INFO, filename=f'Cleaner-{time.time()}.log',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    target = [target] if isinstance(target, str) else target
    
    for t in target:
        logging.info(f'cleaning {t}')
        for root, _, files in os.walk(t):
            if 'JYLmaxRcutJointConvTestDriver.py' in files:
                pporb = os.path.join(root, find_pporb_folder(root))
                logging.info(f'cleaning {pporb} in {root}')
                if os.path.exists(pporb):
                    shutil.rmtree(pporb)
                    logging.info(f'cleaned {pporb} in {root}')
                else:
                    logging.warning(f'{pporb} not found in {root}')
        logging.info(f'cleaned {t}')
        # zip the t
        logging.info(f'compressing {t}')
        fn = compress(t)
        logging.info(f'compressed {t} to {fn}')
        # remove the folder
        logging.info(f'removing {t}')
        shutil.rmtree(t)
        logging.info(f'removed {t}')
        
    logging.shutdown()

    return None

if __name__ == '__main__':
    main(['JYLmaxRcutJointConvTest-Al', 'JYLmaxRcutJointConvTest-B',  'JYLmaxRcutJointConvTest-Be',
          'JYLmaxRcutJointConvTest-C' , 'JYLmaxRcutJointConvTest-Ca', 'JYLmaxRcutJointConvTest-F',
          'JYLmaxRcutJointConvTest-K' , 'JYLmaxRcutJointConvTest-Mg', 'JYLmaxRcutJointConvTest-N',
          'JYLmaxRcutJointConvTest-Na', 'JYLmaxRcutJointConvTest-P',  'JYLmaxRcutJointConvTest-S',
          'JYLmaxRcutJointConvTest-Si'])