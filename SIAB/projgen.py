'''
this module is from the ABACUS source code
source/module_base/projgen.cpp

Concepts
--------
ABACUS generate the projector by truncating the atomic basis, then
do smooth on the truncated basis to get the projector. The projector
should satisfy the following criteria:
1. the projector should be smooth up to the second order at the cutoff
    radius
2. the projector should be normalized
3. the projector should have the most similarity to the atomic basis

Mathematically, the projector |p> can be expressed as:
<r|p> = p(r) = a(r) * [1 - exp(-(r - rc)^2/2/sigma^2)]
where a(r) is the atomic basis, rc is the cutoff radius, sigma is the
smooth parameter, which will be determined in the optimization problem

The optimization problem is designed as:
minimize the kinetic spillage between the atomic basis and the
projector:

Sk = <grad(p - a)|grad(p - a)>
   = <p - a | T | p - a>
   
Manual
------
there are many flags needed to set:
-i, --input
    the input orb file. This is compulsory
-p, --prefix
    the prefix of the output files. This is optional, has the default
    value as `projgen`
-o, --output
    the output projector file. This is optional, if not specified, 
    find the output file in the same directory as the input file. 
    There will be a file with extension like `.proj`
-r, --onsite-radius
    the cutoff radius
-l, --log
    the log file, if not specified, it is okay, it will be named as
    `projgen.log`
-j, --jzeta
    the index of the zeta function to use. This is optional, if not
    specified, use the first zeta function for all the angular momenta.
    This is the case that ABACUS DFT+U or Deltaspin is used. The index
    can also be negative, which means the index is counted from the last
    zeta function. For example, -1 means the last zeta function. NOTE:
    the index is the global index, not the local index. For example
    the In_gga_10au_100Ry_3s3p3d2f.orb, the first d corresponds the the
    index 6, the second f corresponds to the index 7, and so on.
-m, --mode
    the mode how the generated projector write to file. This is optional,
    can be `new` or `update`. If `new`, only the generated projector will
    be written to the output file. If `update`, the generated projector
    will be written along with all other zeta functions not augmented.
    The default value is `new`.

Examples
--------
projgen -i Si_gga_10au_20Ry_1s1p.orb -r 3 -j 0
# generate the projector from the Si_gga_10au_20Ry_1s1p.orb with the cutoff
# radius 3, and use the first zeta function of s orbital.
'''
# in-built modules
import re
import os
import argparse
import logging
import unittest

# third-party modules
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import simps

# local modules
from SIAB.driver.control import OrbgenAssert

def parse_orbpat(forb):
    pat = r'([A-Z][a-z]?)_(\S+)_(\d+au)_(\d+Ry)_((\d+[spdfghi])+)\.orb'
    m = re.match(pat, os.path.basename(forb))
    OrbgenAssert(m, f'{forb} does not match the pattern {pat}')
    key = ['elem', 'xc', 'rcut', 'ecut', 'zetapat']
    return dict(zip(key, m.groups()))

def integer_once_possible(num):
    '''convert a float to int if possible'''
    return int(num) if abs(int(num) - float(num)) < 1e-6 else float(num)

def nzetamap(nzeta):
    '''map from the global index to the local index'''
    return [(l, iz) for l, nz in enumerate(nzeta) for iz in range(nz)]

def init():
    '''read-in the input parameters'''
    helpmsg = 'generate the projector from the atomic basis by \
augmenting the Gaussian smooth function'

    parser = argparse.ArgumentParser(description=helpmsg)
    parser.add_argument('-i', '--input', required=True, 
                        help='input orb file')
    parser.add_argument('-p', '--prefix', default='projgen', 
                        help='prefix of the output files')
    parser.add_argument('-o', '--output', default=None, 
                        help='output projector file')
    parser.add_argument('-r', '--onsite-radius', type=float, required=True, 
                        help='cutoff radius')
    parser.add_argument('-l', '--log', 
                        help='log file')
    parser.add_argument('-j', '--jzeta', type=int, default=None, 
                        help='index of zeta function to use. This index is the global one,\
for example the In_gga_10au_100Ry_3s3p3d2f.orb, the first d corresponds the the index 6,\
the second f corresponds to the index 7, and so on.')
    parser.add_argument('-m', '--mode', default='new',
                        help='mode of the projector generation, can be new or update')
    args = parser.parse_args()
    
    # set
    forb = args.input
    orbpat = parse_orbpat(forb)
    _, _, _, _, zetapat = orbpat.values()
    
    OrbgenAssert(os.path.exists(forb), f'{forb} does not exist',
                 FileNotFoundError)
    prefix = args.prefix
    rc = integer_once_possible(args.onsite_radius)
    fout = args.output
    if fout is None:
        fout = forb.replace('.orb', f'.{rc}au.proj')
    flog = args.log if args.log else f'{prefix}.log'
    izeta = args.jzeta
    if izeta is None:
        pat = r'(\d+)([spdfghi])'
        m = re.findall(pat, zetapat)
        nzeta = np.array([int(i) for i, _ in m])
        izeta = [n - 1 for n in nzeta]
        nzeta_accum = [0] + np.cumsum(nzeta)[:-1].tolist()
        izeta = [nz0 + iz for nz0, iz in zip(nzeta_accum, izeta)]
    elif izeta < 0:
        pat = r'(\d+)([spdfghi])'
        m = re.findall(pat, zetapat)
        nzeta = np.sum([int(i) for i, _ in m])
        izeta = nzeta + izeta
    izeta = [izeta] if not isinstance(izeta, list) else izeta
    
    mode = args.mode
    OrbgenAssert(mode in ['new', 'update'], 
                 f'{mode} is not a valid mode, should be new or update')
    
    return {'forb': forb, 'fout': fout, 'rc': rc, 'flog': flog, 
            'izeta': izeta, 'mode': mode}

def driver(param):
    '''main function'''
    from SIAB.spillage.orbio import read_nao, write_nao    
    # read the atomic basis
    nao = read_nao(param['forb'])
    nzeta = [len(z) for z in nao['chi']]
    # sort the index by l then by iz
    tmp = sorted([nzetamap(nzeta)[i] for i in param['izeta']], key=lambda x: x[0])
    tmp = sorted(tmp, key=lambda x: x[1])
    
    chi = [nao['chi'][l][iz] for l, iz in tmp]
    
    # generate the projector
    r = np.linspace(0, nao['rcut'], int(nao['rcut'] / nao['dr']) + 1)
    p = [smoothgen(c, r, param['rc']) for c in chi]
    
    # write the projector
    base = [[] if param['mode'] == 'new' else nao['chi'][l] for l, _ in enumerate(nzeta)]
    if param['mode'] == 'update':
        for jz, (l, iz) in enumerate(tmp):
            base[l][iz] = p[jz]
    else:
        for jz, (l, iz) in enumerate(tmp):
            base[l].append(p[jz])
    nao['chi'] = base
    
    write_nao(param['fout'], nao['elem'], nao['ecut'], nao['rcut'], 
              nao['nr'], nao['dr'], nao['chi'])
    return param['fout']

def smoothgen(chi, r, rc):
    '''
    generate the projector from the atomic basis by augmenting the
    Gaussian smooth function, whose sigma will be determined by
    minimizing the kinetic spillage between the atomic basis and
    the projector
    
    Parameters
    ----------
    chi: np.ndarray
        the atomic basis
    r: np.ndarray
        the radial grid
    rc: float
        the cutoff radius
        
    Returns
    -------
    p: np.ndarray
        the projector
    '''
    OrbgenAssert(len(chi) == len(r), 
        'the length of chi and r should be the same')
    OrbgenAssert(rc <= r[-1], 
        'the cutoff radius should be smaller than the maximum radius')
    
    logging.info(f'smoothgen is called with rc = {rc}')
    
    def smooth(sigma, r, rc):
        '''the smooth function'''
        g = 1 - np.exp(-(r-rc)**2 / (2 * sigma**2))
        g[r >= rc] = 0
        return g
    def norm(f, r):
        '''the norm of a radial function'''
        return np.sqrt(simps(f**2 * r**2, x=r))
    def proj(f, sigma, r, rcut, normalize=True):
        '''the projector'''
        p = f * smooth(sigma, r, rcut)
        return p / norm(p, r) if normalize else p
    
    def kin_spill(sigma):
        '''the objective function: <grad(p - chi)|grad(p - chi)>'''
        p = proj(chi, sigma, r, rc)
        return simps(np.gradient(p - chi, r) ** 2 * r**2, x=r)
    
    res = minimize(kin_spill, x0=1.0, method='L-BFGS-B', bounds=[(0.1, np.inf)])
    logging.info(f'smooth Gaussian function gets sigma = {res.x[0]}')
    return proj(chi, res.x[0], r, rc)

def main():
    param = init()
    # start logging
    logging.basicConfig(filename=param['flog'], level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('projgen toolkit activated.')
    f = driver(param)
    from SIAB.spillage.plot import plot_orbfile
    plot_orbfile(f, save=os.path.join(os.path.dirname(f), 'projgen.png'))
    
    logging.info('projgen toolkit finished.')
    logging.shutdown()
    print(f'projgen finished. The projector is saved in {f}')

class TestProjgen(unittest.TestCase):
    here = os.path.dirname(__file__)
    forb = os.path.join(here, 'spillage', 'testfiles', 'Si_gga_10au_20Ry_1s1p.orb')
    
    def test_parse_orbpat(self):
        forb = 'Si_gga_10au_20Ry_1s1p.orb'
        orb = parse_orbpat(forb)
        self.assertEqual(orb['elem'], 'Si')
        self.assertEqual(orb['xc'], 'gga')
        self.assertEqual(orb['rcut'], '10au')
        self.assertEqual(orb['ecut'], '20Ry')
        self.assertEqual(orb['zetapat'], '1s1p')     
                          
    @unittest.skip('skip the run of the integrated test')
    def test_smoothgen(self):
        import matplotlib.pyplot as plt
        from SIAB.spillage.orbio import read_nao
    
        nao = read_nao(self.forb)
        
        r = np.linspace(0, nao['rcut'], int(nao['rcut'] / nao['dr']) + 1)
        chi = -np.array(nao['chi'][1][0])

        rc = 3
        
        p = smoothgen(chi, r, rc)
        
        fig, ax = plt.subplots()
        ax.plot(r, chi, label='chi')
        ax.plot(r, p, label='p')
        ax.legend()
        ax.set_xlim(0, nao['rcut'])
        ax.set_ylim(0, 1.0)
        plt.savefig('projgen.png')
        
if __name__ == '__main__':
    # unittest.main(exit=False)
    main()
