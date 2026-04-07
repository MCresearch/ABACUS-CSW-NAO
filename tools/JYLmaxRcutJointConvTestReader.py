import re
import unittest
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import time

def _parse_abacus_orb(forb):
    '''return the information of orbital from the file name'''
    pat = r'([A-Z][a-z]?)_(gga)_([0-9]+)au_([0-9]+(\.[0-9]+)?)Ry_((\d+[spdfghi])+)\.orb'
    def _parse_orb_conf(conf):
        confpat = r'(\d+)([spdfghi])'
        result = re.findall(confpat, conf)
        if result:
            return [int(n) for n, s in result]
        else:
            return []

    m = re.match(pat, forb)
    if m:
        out = dict(zip(['elem', 'xc', 'rcut', 'ecut'], [m.group(i) for i in range(1, 5)]))
        out['nzeta'] = _parse_orb_conf(m.group(6))
        return out
    else:
        return dict()

def read_energy_from_runninglog(fn):
    '''
    read the final energy from `!FINAL_ETOT_IS` in the running_scf.log
    '''
    try:
        with open(fn) as f:
            lines = [line.strip() for line in f.readlines()]
        lines = [line for line in lines if line]
        lines = [line for line in lines if line.startswith('!')]
        if len(lines) > 1: 
            # means there is at least one line like `!! convergence has not been achieved @_@`
            print(f'warning: !! convergence has not been achieved @_@ in {fn}')
        return float(lines[-1].split()[-2])
    except FileNotFoundError:
        print(f'{fn} not found')
        return None
    except IndexError:
        print(f'{fn} has no energy')
        return None
    except ValueError:
        print(f'{fn} has no energy')
        return None

def read_bond_length_and_basis_from_descriptor(fn):
    '''return in Angstrom unit'''
    try:
        with open(fn) as f:
            desc = json.load(f)
        bl = abs(desc['Cell']['coords'][0][0] - desc['Cell']['coords'][1][0])
        basis = desc['DFTParamSet']['basis_type']
        pp = os.path.basename(desc['AtomSpecies'][0]['pp'])
        # if 'FR' in pp:
        #     return dict()
        orb = desc['AtomSpecies'][0]['nao']
        orb = 'invalid' if orb is None else os.path.basename(orb)
        if orb == 'invalid' and basis == 'lcao':
            raise ValueError('lcao basis with invalid nao')
        return dict(zip(['bl', 'basis', 'orb'], [bl, basis, orb]))
    except FileNotFoundError:
        print(f'{fn} not found')
        return dict()

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

def read_all(path, walk = False):
    '''read all folders in path, gather data'''
    folders = _abacus_folder(path, walk)
    data = [(f, 
             read_energy_from_runninglog(os.path.join(f, f'OUT.{suffix}', 'running_scf.log')), 
             *read_bond_length_and_basis_from_descriptor(os.path.join(f, 'description.json')).values()) 
            for f, suffix in folders]
    
    # filter out None
    return [d for d in data if all(d)]

def _colors(n):
    '''generate n colors'''
    return plt.cm.rainbow(np.linspace(0, 1, n))

def parse(target, walk = False):
    '''parse the data from the target folder
    
    Parameters
    ----------
    target: str or list of str
        the target folder or a list of target folders
    walk: bool
        whether to walk through the target folder
    
    Returns
    -------
    pw: list of list of float
        the bond length and energy of pw
    lcao: dict of list of list of float
        the bond length and energy of lcao, key is the file name of orbital
    '''

    pw = [[], []]
    lcao = {}
    target = [target] if isinstance(target, str) else target
    assert all([os.path.exists(t) for t in target])
    for t in target:
        for _, e, bl, basis, forb in read_all(t, walk):
            if basis == 'pw':
                pw[0].append(bl)
                pw[1].append(e)
            else:
                assert basis == 'lcao'
                lcao.setdefault(forb, [[], []])
                lcao[forb][0].append(bl)
                lcao[forb][1].append(e)
    # sort the key of lcao
    lcao = {k: lcao[k] for k in sorted(lcao.keys(), key=lambda forb: int(forb.split('_')[2].replace('au', '')))}

    return pw, lcao

def main(target, 
         walk = False, 
         unit = 'kcal/mol', 
         logscale = True,
         dump_prefix = None, 
         dump_note = None):
    '''the main function
    
    Parameters
    ----------
    target: str|list
        the file that contains the data or the folder that contains the data.
        can also be a list of folders
    walk: bool
        whether to walk through the target folder
    unit: str
        the unit of energy, default is 'kcal/mol'
    logscale: bool
        whether to use log scale for y axis
    dump_prefix: str
        the prefix of the dumped file
    dump_note: str
        the extra note to be added to the dumped file
    '''
    pw, lcao = None, None
    if isinstance(target, str):
        if target.endswith('.json'):
            pw, lcao = load(target)
    if not all([pw, lcao]):
        # fall back to the parse function
        pw, lcao = parse(target, walk)
        fn = f'JYLmaxRcutJointConvTest'
        if dump_prefix is not None:
            fn += '-' + dump_prefix
        fn += f'@{time.strftime("%Y%m%d%H%M%S")}.json'
        dump(pw, lcao, fn, note = dump_note)

    plot(*_customized_sort(pw, lcao), unit = unit, logscale = logscale)

def dump(pw, lcao, fn, note = None):
    '''dump the data to json file'''
    # lcao = {k: v.tolist() for k, v in lcao.items()}
    orbs = [_parse_abacus_orb(k) for k in lcao.keys()]
    # check if all orbs have uniform elem
    if len(set([o['elem'] for o in orbs])) != 1:
        raise ValueError('the orbs have different elements')

    data = {'pw': pw, 'lcao': lcao, 'elem': orbs.pop()['elem']}
    # add note if any
    if note is not None:
        data['note'] = note

    with open(fn, 'w') as f:
        json.dump(data, f)
    return fn

def load(fn):
    '''load the data from json file'''
    with open(fn) as f:
        data = json.load(f)
    # pw = data['pw']
    # lcao = {k: np.array(v) for k, v in data['lcao'].items()}
    # return pw, lcao
    return data['pw'], data['lcao']

def _customized_reshape(y):
    '''manually reshape the y from the 1d list of tuple (lmax, rcut, e)
    to be the one indexed by [il][ir]. The il means the index of angular momentum
    , and ir means the index of rcut
    
    Parameters
    ----------
    y: list of tuple
        the 1d list of tuple (lmax, rcut, e)
    
    Returns
    -------
    y: list of np.array
        the reshaped y
    '''
    lmax = sorted(list(set([l for l, _, _ in y])))
    temp = [[(r, e) for l, r, e in y if l == l_] for l_ in lmax]
    temp = [[e for _, e in sorted(t)] for t in temp]
    return [np.array(t) for t in temp]

def _customized_sort(pw, lcao):
    '''this function is for preprocessing the data in three aspects:
    1. fill the missing bond length data points with None
    2. delete the duplicate bond length data points
    3. sort the bond length data points
    
    Parameters
    ----------
    pw: list of list of float
        the bond length and energy of pw
    lcao: dict of list of list of float
        the bond length and energy of lcao, key is the file name of orbital
    
    Returns
    -------
    pw: np.array
        the bond length and energy of pw
    lcao: dict of np.array
        the bond length and energy of lcao, key is the file name of orbital
    '''
    # check the integrity of the data, if there is missing datapoints, fill value as None
    bl = set(pw[0]).union(*[set(v[0]) for v in lcao.values()]) # the union of bond lengths
    for k, v in lcao.items():
        for b in bl:
            if b not in v[0]:
                v[0].append(b)
                v[1].append(None)
    for b in bl:
        if b not in pw[0]:
            pw[0].append(b)
            pw[1].append(None)

    # delete duplicate bond length data points
    pw = np.array(pw)
    idx = np.unique(pw[0], return_index=True)[1]
    pw = pw[:, idx]
    for k, v in lcao.items():
        lcao[k] = np.array(v)
        idx = np.unique(lcao[k][0], return_index=True)[1]
        lcao[k] = lcao[k][:, idx]

    # sort by bond length
    pw = pw[:, np.argsort(pw[0])]
    for k, v in lcao.items():
        lcao[k] = lcao[k][:, np.argsort(lcao[k][0])]

    return pw, lcao

def plot(pw, lcao, unit = 'eV', logscale = False):
    '''compare the JY lmax-rcut joint convergence test taking the pw calculation
    result as reference.
    
    Parameters
    ----------
    pw: np.array
        the bond length and energy of pw: [[bl], [e]]
    lcao: dict of np.array
        the bond length and energy of lcao, key is the file name of orbital, value
        is the np.array of [[bl], [e]]
    unit: str
        the unit of energy, default is 'eV'
    logscale: bool
        whether to use log scale for y axis
    '''
    factor = {'eV': 1, 'Ha': 1/27.2114, 'meV': 1000, 'Ry': 1/13.6057,
              'kcal/mol': 23.0605, 'kJ/mol': 96.4853}
    fontsize = 12

    # convert unit
    pw[1] = np.array([p*factor[unit] if p is not None else None for p in pw[1]])
    for k, v in lcao.items():
        lcao[k][1] = np.array([p*factor[unit] if p is not None else None for p in v[1]])

    # plot, the last one plots the averaged energy over bond lengths
    fig, ax = plt.subplots(2, int(np.ceil((len(pw[0]) + 1)/2)), figsize=(20, 12), squeeze=False)

    for i in range(len(pw[0])): # for each bond length...
        # get the energy data of pw
        y0 = pw[1][i]
        # get the energy data of each lcao at present bond length
        # different forb will have different lmax and rcut
        y = [(*_parse_abacus_orb(k).values(), v[1][i]) for k, v in lcao.items()]
        y = [(len(nzeta) - 1, float(rcut), e) for _, _, rcut, _, nzeta, e in y]
        lmax = sorted(list(set([l for l, _, _ in y])))
        rcut = sorted(list(set([r for _, r, _ in y])))

        # sort the lcao data by lmax, then rcut
        y = sorted(y, key=lambda x: (x[0], x[1]))

        # then make y can be indexed by [lmax][rcut]
        y = _customized_reshape(y)

        # each lmax (with different rcut) will form one curve
        figx, figy = i % 2, i // 2
        for j, l in enumerate(lmax):
            temp = [(r, yji - y0) for r, yji in zip(rcut, y[j]) if yji is not None]
            if len(temp) == 0:
                print(f'No data for lmax={l} at bond length {pw[0][i]:.2f} Angstrom')
                continue
            x, y_ = zip(*temp)
            ax[figx, figy].plot(x, y_, 'o-', label=f'lmax={l}')
            ax[figx, figy].grid(True)
            if logscale:
                ax[figx, figy].set_yscale('log')
                #ax[figx, figy].set_ylim([1e-2, 1e2])
            ax[figx, figy].set_xlabel('rcut (au)', fontsize = fontsize)
            ax[figx, figy].set_ylabel(f'Energy ({unit})', fontsize = fontsize)
            ax[figx, figy].legend()
            ax[figx, figy].set_title(f'Bond Length: {pw[0][i]:.2f} Angstrom', fontsize = fontsize)
    
    # averaging the error to PW over all bond lengths
    cal_error = lambda x, y: None if any([x is None, y is None]) else x - y
    cal_mean = lambda x: np.mean([i for i in x if i is not None])
    cal_stddev = lambda x: np.std([i for i in x if i is not None])
    y = {k: [cal_error(i, j) for i, j in zip(v[1], pw[1])] for k, v in lcao.items()}
    #y = {k: (cal_mean(v), cal_stddev(v)) for k, v in y.items()}
    y = [(len(_parse_abacus_orb(k)['nzeta']) - 1, float(_parse_abacus_orb(k)['rcut']), d) for k, d in y.items()]
    lmax = sorted(list(set([l for l, _, _ in y])))
    rcut = sorted(list(set([r for _, r, _ in y])))
    y = sorted(y, key=lambda x: (x[0], x[1]))
    y = _customized_reshape(y)
    for j, l in enumerate(lmax):
        ymean = [cal_mean(yji) for yji in y[j]]
        ystddev = [cal_stddev(yji) for yji in y[j]]
        temp = [(r, yji) for r, yji in zip(rcut, ymean) if yji is not None]
        x, y_ = zip(*temp)
        ax[-1, -1].errorbar(x, y_, yerr=ystddev, fmt='o-', label=f'lmax={l}')
        ax[-1, -1].grid(True)
        if logscale:
            ax[-1, -1].set_yscale('log')
            ax[-1, -1].set_ylim([1e-2, 1e2])
        ax[-1, -1].set_xlabel('rcut (au)', fontsize = fontsize)
        ax[-1, -1].set_ylabel(f'Energy ({unit})', fontsize = fontsize)
        ax[-1, -1].legend()
        ax[-1, -1].set_title(f'Averaged over all bond lengths', fontsize = fontsize)

    plt.tight_layout()
    # return fig
    plt.savefig('JYLmaxRcutJointConvTest.png')

class TestJYLmaxConvergence(unittest.TestCase):

    def test_parse_abacus_orb(self):
        self.assertEqual(_parse_abacus_orb('H_gga_10au_10.0Ry_1s.orb'), 
                         {'elem': 'H', 'xc': 'gga', 'rcut': '10', 'ecut': '10.0', 'nzeta': [1]})
        self.assertEqual(_parse_abacus_orb('H_gga_10au_10.0Ry_1s2p.orb'), 
                         {'elem': 'H', 'xc': 'gga', 'rcut': '10', 'ecut': '10.0', 'nzeta': [1, 2]})
        self.assertEqual(_parse_abacus_orb('H_gga_10au_10.0Ry_1s2p3d.orb'), 
                         {'elem': 'H', 'xc': 'gga', 'rcut': '10', 'ecut': '10.0', 'nzeta': [1, 2, 3]})
        self.assertEqual(_parse_abacus_orb('H_gga_10au_10.0Ry_1s2p3d4f5g6h7i.orb'), 
                         {'elem': 'H', 'xc': 'gga', 'rcut': '10', 'ecut': '10.0', 'nzeta': [1, 2, 3, 4, 5, 6, 7]})

if __name__ == '__main__':
    unittest.main(exit=False)
    elem = 'C'
    main(target=f'temp.json', 
         walk = True, 
         unit = 'kcal/mol', 
         logscale = True,
         dump_prefix = elem,
         dump_note = '')
