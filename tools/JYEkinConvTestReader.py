import json
import os
import re
import unittest

from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

ETA_FORMULA =  '$\eta = \min_{\omega}{'
ETA_FORMULA += '\\frac{'
ETA_FORMULA +=  '\sum_{n\mathbf{k}}{'
ETA_FORMULA +=   '\\tilde{f}_{n\mathbf{k}}(e_{1, n\mathbf{k}}-e_{2, n\mathbf{k}} + \omega)^2}}'
ETA_FORMULA +=    '{\sum_{n\mathbf{k}}{\\tilde{f}_{n\mathbf{k}}}'
ETA_FORMULA +=   '}'
ETA_FORMULA += '}$'

def read_istate(fn):
    '''
    read the band structure from the file `istate.info` from ABACUS outdir.
    The structure of file is like:
    ```
    BAND Energy(ev) Occupation Kpoint = 1 (0.000000, 0.000000, 0.000000)
    1 0.000000 0.000000
    2 0.000000 0.000000
    ...
    
    BAND Energy(ev) Occupation Kpoint = 2 (0.000000, 0.500000, 0.000000)
    1 0.000000 0.000000
    2 0.000000 0.000000
    ...
    ```
    For nspin 2 case, the file is like:
    ```
    BAND Spin up Energy(ev) Occupation Spin down Energy(ev) Occupation Kpoint = 1 (0.000000, 0.000000, 0.000000)
    1 0.000000 0.000000 0.000000 0.000000 0.000000
    2 0.000000 0.000000 0.000000 0.000000 0.000000
    ...
    ```
    
    Parameters
    ----------
    fn : str
        the path of the file `istate.info`
    
    Returns
    -------
    list[dict]
        a list of dict, each dict stores the kpoint coordinates (`kcoord` -> list[float]), 
        and energy (`ekb` -> list[list[float]]), occupation (`occ` -> list[list[float]]). 
        Both the ekb and occ are indexed firstly by the spin, then by the band index.
    '''
    def datparse(dat):
        '''parse the data'''
        nbands = int(np.max(dat[:, 0]))
        nkpts = int(len(dat) / nbands)
        dat = dat.reshape(nkpts, nbands, -1) # indexed by [ik][ib]
        occ = dat[:, :, 2::2]
        ekb = dat[:, :, 1::2]
        return occ, ekb
    def headerparse(header):
        '''parse the header'''
        kcoord_ = r'\(\s*-?\d+(\.\d+)?\s+-?\d+(\.\d+)?\s+-?\d+(\.\d+)?\s*\)'
        return [float(x) for x in re.search(kcoord_, header).group()[1:-1].split()]
    try:
        with open(fn) as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f'{fn} not found')
        return None
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    # it is not wise to rely on the natural language, instead, we use the number of columns
    # to determine the structure of the file
    nspin = len(lines[1].split()) // 2
    assert nspin in [1, 2], 'nspin should be 1 or 2'
    
    pat_ = r'\s*(\d+)\s+(-?\d+(\.\d+)?)\s+(\d+(\.\d+)?)(\s+\d+(\.\d+)?)?'
    headers = [headerparse(l) for l in lines if not re.match(pat_, l)]
    
    data = np.array([[float(x) for x in l.split()] for l in lines 
                      if re.match(pat_, l)])
    data = datparse(data)   
    
    return [{'kcoord': k, 
             'occ': o.T.tolist(), 
             'ekb': e.T.tolist()} 
            for k, o, e in zip(headers, *data)]

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

def read_bond_length_and_ekin_from_descriptor(fn):

    try:
        with open(fn) as f:
            desc = json.load(f)
        pp = os.path.basename(desc['AtomSpecies'][0]['pp'])
        if 'FR' in pp:
            return dict()
        bl = abs(desc['Cell']['coords'][0][0] - desc['Cell']['coords'][1][0])
        orb = desc['AtomSpecies'][0]['nao']
        orb = 'invalid' if orb is None else os.path.basename(orb)
        if orb == 'invalid':
            raise ValueError('lcao basis with invalid nao')
        ekin = float(orb.split('_')[3].replace('Ry', ''))
        return {'bl': bl, 'ekin': ekin}
    except FileNotFoundError:
        print(f'{fn} not found')
        return dict()

def read_ekin_from_descriptor(fn):

    try:
        with open(fn) as f:
            desc = json.load(f)
        pp = os.path.basename(desc['AtomSpecies'][0]['pp'])
        # if 'FR' in pp:
        #     return dict()
        orb = desc['AtomSpecies'][0]['nao']
        orb = 'invalid' if orb is None else os.path.basename(orb)
        if orb == 'invalid':
            raise ValueError('lcao basis with invalid nao')
        ekin = float(orb.split('_')[3].replace('Ry', ''))
        return {'ekin': ekin}
    except FileNotFoundError:
        print(f'{fn} not found')
        return dict()

def parse(root, iterative = False):
    '''
    iteratively read all the data from the given root folder.
    
    Parameters
    ----------
    root : str
        the path of the root folder
    iterative : bool
        if True, walk the folder and find all the subfolders. 
        If False, only read the folders in the root folder.
        Default is False.
    
    Returns
    -------
    list[dict]
        a list of dict, each dict stores the `folder`, `energy`, `ekin` and `istate`
    '''
    folders = _abacus_folder(root, iterative)

    data = [(f, 
             read_energy_from_runninglog(os.path.join(f, f'OUT.{suffix}', 'running_scf.log')),
             *read_ekin_from_descriptor(os.path.join(f, 'description.json')).values(),
             read_istate(os.path.join(f, f'OUT.{suffix}', 'istate.info')))
            for f, suffix in folders]

    return [dict(zip(['folder', 'energy', 'ekin', 'istate'], d))
            for d in data if all(d[1:])]

def cal_istate_dist(istate1, istate2, ibands=None, occ_wt=False):
    '''calculate the `eta` value from the two istate (collection of energy and occupation)
    
    Parameters
    ----------
    istate1 : list[dict]
        the istate that returned by the function `read_istate`
    istate2 : list[dict]
        similar to istate1
    ibands : np.ndarray|None
        the band indexes to be used.
        If None, then all the bands will be used.
        If a list of integers, it will be understood as the index of the bands
    occ_wt : bool
        Whether use the occupation as the weight of bands, if not, will set
        the all bands equally weighted.

    Returns
    -------
    float
        the `eta` value
    '''
    # sanity check
    nk = len(istate1)
    assert nk == len(istate2), 'the number of kpoints should be the same'
    nspin = len(istate1[0]['ekb'])
    assert all([len(istate1k['ekb']) == nspin for istate1k in istate1]), \
              'the number of spins should be the same'
    assert all([len(istate2k['ekb']) == nspin for istate2k in istate2]), \
              'the number of spins should be the same'
    # it is not always that two istates have the same number of bands,
    # so we need to find the minimum number of bands and truncate the bands
    nbands_min = min([len(istatek['ekb'][i]) 
                      for istate in [istate1, istate2]
                      for i in range(nspin) 
                      for istatek in istate])
    ibands = ibands[ibands <= nbands_min] if ibands is not None\
        else np.arange(nbands_min)

    # flatten the e1 from [ik][ispin][iband] to [ik][j]
    e1   = np.array([istate1k['ekb'] for istate1k in istate1])[:,:,ibands].flatten()
    occ1 = np.array([istate1k['occ'] for istate1k in istate1])[:,:,ibands].flatten()\
        if occ_wt else np.ones_like(e1)
    e2 =   np.array([istate2k['ekb'] for istate2k in istate2])[:,:,ibands].flatten()
    occ2 = np.array([istate2k['occ'] for istate2k in istate2])[:,:,ibands].flatten()\
        if occ_wt else np.ones_like(e2)
    
    # geometric mean of occ1 and occ2 yields occ that also indexed by k then band
    occ = np.sqrt(occ1 * occ2)
    def eta(omega):
        '''the function to be minimized'''
        return np.sqrt(np.sum((e1 - e2 + omega)**2*occ)/np.sum(occ))
    
    return minimize(eta, 0).fun

def _pick_bands(ekb, nzeta, delta=1e-4):
    '''
    pick-up enough band-indexes for each angluar momentum specified by `nzeta`.
    For example, if nzeta = [2, 3], then we need to pick up 2+3*3=5 bands, in
    which two s-orbital and three p-orbital.
    
    Parameters
    ----------
    ekb : list[float]
        the energy of the bands
    nzeta : list[int]
        the number of bands for each angular momentum
        The order is s, p, d, f, g, h, i, j, k, l...
        
    Returns
    -------
    list[int]
        the band indexes
    '''
    # drop the nzeta for l >= 4 because it is cannot be generated from a 
    # atomic calculation
    nzeta = nzeta[:4] if len(nzeta) > 4 else nzeta
    
    # first round all data to the resolution of delta
    ekb = np.round(ekb, decimals=int(-np.log10(delta)))
    
    # we identify the band's angular momentum by the degree of the degeneracy
    # for example, if the band is 3-fold degenerate, then it is a p-orbital...
    e, ibands, degen = np.unique(ekb, return_index=True, return_counts=True)
    
    # assert that all degen should be odd number except the last one
    assert all([d % 2 == 1 for d in degen[:-1]]), \
        f'the band should be odd degenerate, but got:\n{degen[:-1]}\n' \
        f'change the `delta` and retry.'
    
    # pick
    temp = [[ib for ib, mul in zip(ibands, degen) if mul == (2*l+1)][:nz] 
            for l, nz in enumerate(nzeta)]
    return [[list(range(i0l, i0l+(2*l+1))) for i0l in i0ls] 
            for l, i0ls in enumerate(temp)]

def calculate(data, 
              unit='kcal/mol', 
              nzeta=None, 
              ibands=None,
              delta=1e-4,
              occ_wt=False):
    '''
    according to the given data, calculate the convergence curve
    
    Parameters
    ----------
    data : list[tuple]
        the data returned by the function `read`
    unit : str
        the unit of the energy difference. 
        The default is 'kcal/mol'. Other options are 'eV', 'Ha', 'meV', 'Ry', 'kJ/mol'.
    ibands : list[int]|Literal['all']|None
        the index of the bands to be used. 
        If None or 'all', then all the bands will be used.
        If a list of integers, it will be understood as the number of bands
        for each angular momentum.
    
    Returns
    -------
    tuple
        a tuple of two lists, the first list is the kinetic energy,
        the second list is the energy difference between the two istate.
    '''
    factor = {'eV': 1, 'Ha': 1/27.2114, 'meV': 1000, 'Ry': 1/13.6057,
              'kcal/mol': 23.0605, 'kJ/mol': 96.4853}

    # sort the data by the kinetic energy
    ekin = [d['ekin'] for d in data]
    data = [data[i] for i in np.argsort(ekin)]

    # eks = [d[1] for d in data] # ground-state energy is little of meaning
    istates = [d['istate'] for d in data] # ordered in an ascending order of kinetic energy

    # band selection, sanity check
    assert len([x for x in [ibands, nzeta] if x is not None]) <= 1, \
        'ibands and nzeta setting conflict, set one of them to None'
    if nzeta is not None and ibands is None:
        print('JYEkinConvTest: calculate the `ibands` with given `nzeta`')
        assert all([len(istate) == 1 for istate in istates]), \
            'pick band runtime error: the band structure should be single kpoint'
        ibands = _pick_bands(istates[-1][0]['ekb'], # use the assumed-to-be-converged
                             nzeta, delta=delta)
        # flatten the ibands returned by func _pick_bands that indexed by [l][n][i]
        ibands = np.array([i for l in ibands for n in l for i in n])
    else:
        print('JYEkinConvTest: `ibands` read-in or set default as `all`.')
        ibands = np.array(ibands) if ibands in ['all', None] \
            else np.array(len(istates[-1][0]['ekb']))      

    return [d['ekin'] for d in data], \
           [cal_istate_dist(istate, istates[-1], 
                            ibands=ibands,
                            occ_wt=occ_wt) * factor[unit] \
               for istate in istates]

def main(src, 
         iterparse=False, 
         ibands=None, 
         nzeta=None,
         occ_wt=False):
    '''
    Read the result of ecutjy convergence test, plot the convergence curve
    and save the result to a file.
    
    Parameters
    ----------
    src : str
        the path of the folder or file to be read
    iterparse : bool
        whether to read the folder recursively
        Default is False.
    ibands : list[int]|Literal['all']|None
        the index of the bands to be used.
        If None or 'all', then all the bands will be used.
        If a list of integers, it will be understood as the number of bands.
    nzeta : list[int]|None
        the number of orbitals for each angular momentum.
        The order is s, p, d, f, g, h, i, j, k, l...
    occ_wt : bool
        Whether use the occupation as the weight of bands, if not, will set
        the all bands equally weighted.
    
    Returns
    -------
    tuple
        a tuple of two lists, the first list is the kinetic energy,
        the second list is the energy difference between the two istate.
    '''
    # sanity check: whether the file exists
    assert os.path.exists(src), f'{src} not found'
    
    # data file
    if os.path.isfile(src) and src.endswith('.json'):
        with open(src) as f:
            data = json.load(f)
        return data['ekin'], data['eta']
    # folder to parse
    else:
        assert not all([x is not None for x in [ibands, nzeta]]), \
            'ibands and nzeta setting conflict, set one of them to None'
        fn = os.path.basename(src) if iterparse else os.path.dirname(src)
        fn = os.path.basename(fn) + '.json'
        # directly calculate on what parsed
        ekin, eta = calculate(data=parse(src, iterative=iterparse), 
                              unit='eV', 
                              ibands=ibands, 
                              nzeta=nzeta,
                              delta=1e-3,
                              occ_wt=occ_wt)
        data = {'ekin': ekin, 'eta': eta}
        with open(fn, 'w') as f:
            json.dump(data, f, indent=4)
        return ekin, eta

class TestJYEkinConvergence(unittest.TestCase):

    def test_read_istate(self):
        here = os.path.dirname(__file__)
        fk_nspin1 = os.path.join(here, 'testfiles', 'istate-multik-nspin1.info')
        tmp = read_istate(fk_nspin1)
        self.assertEqual(len(tmp), 8) # 8 kpoints
        for i in range(8):
            self.assertEqual(len(tmp[i]['kcoord']), 3) # k coordinates in 3D
            self.assertEqual(len(tmp[i]['occ']), 1) # nspin 1
            self.assertEqual(len(tmp[i]['ekb']), 1) # nspin 1
            for occ, ekb in zip(tmp[i]['occ'], tmp[i]['ekb']):
                self.assertEqual(len(occ), len(ekb)) # same number of bands
        fk_nspin2 = os.path.join(here, 'testfiles', 'istate-multik-nspin2.info')
        tmp = read_istate(fk_nspin2)
        self.assertEqual(len(tmp), 13) # 13 kpoints
        for i in range(13):
            self.assertEqual(len(tmp[i]['kcoord']), 3)
            self.assertEqual(len(tmp[i]['occ']), 2)
            self.assertEqual(len(tmp[i]['ekb']), 2)
            for occ, ekb in zip(tmp[i]['occ'], tmp[i]['ekb']):
                self.assertEqual(len(occ), len(ekb))
        fgam_nspin1 = os.path.join(here, 'testfiles', 'istate-gamma-nspin1.info')
        tmp = read_istate(fgam_nspin1)
        self.assertEqual(len(tmp), 1) # 1 kpoint
        self.assertEqual(len(tmp[0]['kcoord']), 3)
        self.assertEqual(len(tmp[0]['occ']), 1)
        self.assertEqual(len(tmp[0]['ekb']), 1)
        for occ, ekb in zip(tmp[0]['occ'], tmp[0]['ekb']):
            self.assertEqual(len(occ), len(ekb))

    def test_pick_bands(self):
        ekb = [
             -10.760633729977820,                                       # s
              -4.029649629424734,-4.029649629424306,-4.029649629424306, # p
               0.524574353568484,                                       # s
               2.000478855852826, 2.000478855853168, 2.000487139122785, # d
               2.000487139122785, 2.000487139122785, 
               2.240269791923288, 2.240269791923636, 2.240269791923636, # p
               5.133679744950052, 5.133679744950052, 5.133679744950052, # f
               5.133688032869749, 5.133717700147124, 5.133717700147528,
               5.133717700147528, 
               6.394475293672512, 6.394475293672853, 6.394494496814234, # d
               6.394494496814234, 6.394494496814577, 
               7.150710598856172,                                       # s
               9.466340056149706, 9.466340056149706, 9.466340056149706, # p
               12.13365927597664, 12.13365927597664,12.133659275976640, # f
               12.13367045522659, 12.13369973393518,12.133699733935180,
               12.13369973393518, 
               13.87170628010312, 13.87170628010344,13.871725583900850, # d
               13.87172558390085, 13.87172558390085,
               17.35015105508258,                                       # s
               19.97274689468957, 19.97274689469009,19.972746894690090, # p
               21.70691139071691, 21.70691139071691,21.706911390717280, # f
               21.70691918395755, 21.70692619582389,21.706926195824180,
               21.70692619582418, 
               24.31508108749323, 24.31508108749382,24.315088119147450, # d
               24.31508811914745, 24.31508811914773, 
               30.88631485338270,                                       # s
               33.63817201971364, 33.63817201971410,33.638172019714100, # p
               34.13595137524666, 34.13595137524700,34.135951375247350, # f
               34.13595140449938, 34.13595140449975,34.135951404499750,
               34.13595329037278,
               37.59442465323533]                                       # ?
        out = _pick_bands(ekb, [3, 3, 2, 1]) # the 3s3p2d1f orbital configuration
        self.assertEqual(len(out), 4) # 4 angular momentum
        self.assertEqual(len(out[0]), 3) # 3 s-orbital
        self.assertTrue(all([len(out[0][i]) == 1 for i in range(3)]))
        self.assertEqual(out[0], [[0], [4], [25]])
        self.assertEqual(len(out[1]), 3) # 3 p-orbital
        self.assertTrue(all([len(out[1][i]) == 3 for i in range(3)]))
        self.assertEqual(out[1], [[1, 2, 3], [10, 11, 12], [26, 27, 28]])
        self.assertEqual(len(out[2]), 2) # 2 d-orbital
        self.assertTrue(all([len(out[2][i]) == 5 for i in range(2)]))
        self.assertEqual(out[2], [[5, 6, 7, 8, 9], [20, 21, 22, 23, 24]])
        self.assertEqual(len(out[3]), 1) # 1 f-orbital
        self.assertTrue(all([len(out[3][i]) == 7 for i in range(1)]))
        self.assertEqual(out[3], [[13, 14, 15, 16, 17, 18, 19]])

    def test_calculate_eta(self):
        here = os.path.dirname(__file__)
        fn1 = os.path.join(here, 'testfiles', 'istate-multik-nspin1.info')
        istate1 = read_istate(fn1)
        istate2 = istate1.copy()
        eta = cal_istate_dist(istate1, istate2)
        self.assertEqual(eta, 0)
        
        for istate2k in istate2:
            for i in range(len(istate2k['ekb'][0])):
                istate2k['ekb'][0][i] += 10
        eta = cal_istate_dist(istate1, istate2)
        self.assertEqual(eta, 0)
        
        fn2 = os.path.join(here, 'testfiles', 'istate-multik-nspin2.info')
        istate2 = read_istate(fn2)
        istate1 = istate2.copy()
        for istate2k in istate2:
            for i in range(len(istate2k['ekb'][0])):
                istate2k['ekb'][0][i] += 10
        eta = cal_istate_dist(istate1, istate2)
        self.assertEqual(eta, 0)

if __name__ == '__main__':
    unittest.main(exit=False)
    ekin, eta = main('JYEkinConvTest-Si', 
                     iterparse=True, 
                     nzeta=[3, 3, 2, 1], # Dunning's cc-pVTZ level basis
                     occ_wt=False)
    # plot the convergence curve
    plt.rc('text', usetex=True)
    plt.rc('figure', dpi=300)
    
    plt.plot(ekin[:-1], eta[:-1], ls='-', 
             marker='o', markerfacecolor='white',
             label='2s2p1d')
    plt.xlabel('Kinetic Energy cutoff of spherical wave (Ry)',
               fontsize=14)
    plt.ylabel('Band structure convergence $\eta$ (eV)',
               fontsize=14)
    plt.axhline(2e-2, color='blue', lw=1, ls='--')
    plt.text(x=5, y=2e-2,
             ha='left', va='bottom',
             s='err:not-bad: $\eta < 20$ meV', fontsize=12, color='blue')
    plt.axhline(1e-2, color='red',  lw=1, ls='--')
    plt.text(x=5, y=1e-2,
             ha='left', va='bottom',
             s='err:safe: $\eta < 10$ meV', fontsize=12, color='red')
    plt.title('Band structure convergence test',
               fontsize=14)
    # fix the formula at the top-right corner
    plt.text(x=0.95*plt.xlim()[1], y=0.95*plt.ylim()[1],
             ha='right', va='top',
             s=ETA_FORMULA, fontsize=12,
             bbox=dict(boxstyle='round,pad=0.3', 
                       edgecolor='black', facecolor='white'))
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('EkinConvTest.png')
    plt.close()
