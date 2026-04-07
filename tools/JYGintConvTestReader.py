import json
import os
import matplotlib.pyplot as plt
import numpy as np

def read_energy_from_runninglog(fn):
    try:
        with open(fn) as f:
            lines = [line.strip() for line in f.readlines()]
        return float([line for line in lines if line.startswith('!')][0].split()[-2])
    except FileNotFoundError:
        print(f'{fn} not found')
        return None
    except IndexError:
        print(f'{fn} has no energy')
        return None
    except ValueError:
        print(f'{fn} has no energy')
        return None

def _abacus_folder(path, walk = False):
    '''return the abacus folder and suffix'''
    if walk:
        # the dirname and the suffix
        folders = [(os.path.dirname(f[0]), os.path.basename(f[0]).split('OUT.')[1]) 
                   for f in os.walk(path) if 'OUT.' in f[0]]
    else:
        folders = [(os.path.join(path, f), 'ABACUS') for f in os.listdir(path) 
                   if os.path.isdir(os.path.join(path, f))]
    return folders

def read_ecut_orb_from_descriptor(fn):

    try:
        with open(fn) as f:
            desc = json.load(f)
        pp = os.path.basename(desc['AtomSpecies'][0]['pp'])
        if 'FR' in pp:
            return dict()
        ecut = desc['DFTParamSet']['ecutwfc']
        orb = desc['AtomSpecies'][0]['nao']
        orb = 'invalid' if orb is None else os.path.basename(orb)
        if orb == 'invalid':
            raise ValueError('lcao basis with invalid nao')
        return {'ecut': ecut, 'orb': orb}
    except FileNotFoundError:
        print(f'{fn} not found')
        return dict()

def read_all(target, walk = False):
    
    folders = _abacus_folder(target, walk)
    data = [(f, read_energy_from_runninglog(os.path.join(f, f'OUT.{suffix}', 'running_scf.log')),
             *read_ecut_orb_from_descriptor(os.path.join(f, 'description.json')).values())
            for f, suffix in folders]

    return [d for d in data if all(d[1:])]

def compare(data, unit = 'kcal/mol'):
    factor = {'eV': 1, 'Ha': 1/27.2114, 'meV': 1000, 'Ry': 1/13.6057,
              'kcal/mol': 23.0605, 'kJ/mol': 96.4853}
    reorganized = {}
    for d in data:
        _, e, ecut, orb = d
        reorganized.setdefault(orb, [[], []])
        reorganized[orb][0].append(ecut)
        reorganized[orb][1].append(e)
    for k in reorganized:
        reorganized[k] = np.array(reorganized[k])
        # sort
        idx = np.argsort(reorganized[k][0])
        reorganized[k] = reorganized[k][:, idx]
    # sort the key
    reorganized = dict(sorted(reorganized.items(), key = lambda x: float(x[0].split('_')[3].replace('Ry', ''))))

    for k, v in reorganized.items():
        x, y = v
        label = '$E^{jY}_{kin}$ = ' + k.split('_')[3].replace('Ry', '') + ' Ry'
        plt.plot(x[:-1], np.abs(y - y[-1])[:-1] * factor[unit], 'o-', label = label, markerfacecolor = 'white')
    plt.xlabel('ecutwfc (Ry)')
    plt.ylabel(f'Energy difference ({unit})')
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('GintConvergence-logscale.png')
    # plt.show()

def main(target, walk = False):

    data = read_all(target, walk)
    compare(data)

if __name__ == '__main__':
    main('/root/documents/simulation/abacus/GintConvergenceTestExtended/abacustest-autosubmit-2024-11-20_19-34-40')