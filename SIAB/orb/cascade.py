# in-built modules
import os
import unittest
import logging
import time
from copy import deepcopy
from typing import List, Dict, Optional, Tuple

# third-party modules
import numpy as np

# local modules
from SIAB.driver.control import OrbgenAssert
from SIAB.orb.orb import Orbital, implemented_orbital_models
from SIAB.spillage.spillage import Spillage_pw, Spillage_jy
from SIAB.spillage.spilltorch import SpillTorch_jy, SpillTorch_pw
from SIAB.spillage.listmanip import merge
from SIAB.io.convention import dft_folder
from SIAB.spillage.datparse import read_input_script

def OrbgenCascadeSanitize(initializer: Optional[Dict[str, str]],
                          orbitals: List[Orbital],
                          chkpts: List[int]):
    if initializer is not None:
        OrbgenAssert(isinstance(initializer, List), 
                     'initializer should be a list')
        OrbgenAssert(all(isinstance(i, Dict) for i in initializer),
                     'for each initializer, it should be a dict')
        OrbgenAssert(all('model' in i for i in initializer),
                     'for each initializer, it should have a `model` key')
        OrbgenAssert(all(i['model'] in implemented_orbital_models for i in initializer),
                     'The `model` parameter for each orbital can only be one of '
                     f'the following: {", ".join([f"`{x}`" for x in implemented_orbital_models])} '
                     f'but got: {[i["model"] for i in initializer]}')

    rcuts = list(set([orb.rcut_ for orb in orbitals]))
    OrbgenAssert(len(rcuts) == 1, 'rcut should be the same for all orbitals')
    
    OrbgenAssert(all([i is None or isinstance(i, int) for i in chkpts]),
                 'chkpts should be a list of int or None',
                 TypeError)

class OrbgenCascade:
    '''
    Concepts
    --------
    The OrbgenCascade is a runner for the optimization of orbitals 
    in manner of onion (optimize from the innermost orbital to the
    outermost), let orbitals' optimization tasks form a cascade. 
    '''

    def __init__(self,
                 initializer: Optional[List[Dict[str, str]]],
                 orbitals: List[Orbital],
                 chkpts: List[int],
                 spill_coefs: Tuple = (0, 1),
                 mode: str = 'jy',
                 optimizer: str = 'torch.swats',
                 fix=None):
        '''instantiation of the an orbital cascade
        
        Parameters
        ----------
        initializer : str
            the folder where the initial guess of the coefficients are stored
        orbitals : list[Orbital]
            the list of orbitals forming the cascade
        chkpts : list[int|None]
            the index of its inner shell orbital to be frozen. If None, it
            means a fully optimization
        spill_coefs : tuple[float, float]
            the coefficients for the spillage, the first one is the coefficient
            of the overlap spillage, the second one is the coefficient
            of the arbitrary spillage, the default is (0, 1), which means the
            overlap spillage is not considered
        mode : str
            the mode of the optimization, can be `jy` or `pw`
        optimizer : str
            the optimizer to use, can be `torch.swats`, ..., `scipy.bfgs`
        fix : list[list[list[int]]]|None
            the nested list for all orbitals signing the fixed zeta functions'
            indexes. For example, `fix = [[[0, 1], [2, 3]], [[], [2, 3]]]`
            means: for the first orbital, fix l = 0 the first (indexed by `0`) 
            and second zeta (indexed by `1`) functions, for l = 1, fix the 
            third (`2`) and fourth (`3`) zeta functions. For the second orbital,
            fix the third and fourth for l = 1, l = 0 will be fully relaxed.
        
        Note
        ----
        The combined use of `fix` and `vloc_aux` can be used to directly add g
        orbital of pseudoatom without further optimization: `fix = [[[], [], 
        [], [], [0]]]` with `auxpot = ["path/to/the/pseudopotential/file"]`
        '''
        OrbgenCascadeSanitize(initializer, 
                              orbitals, 
                              chkpts)

        rcuts = list(set([orb.rcut_ for orb in orbitals]))
        self.orbitals_ = orbitals
        self.chkptind_ = chkpts
        
        # initiaizer settings
        self.initializer_ = initializer
        # unique folders...avoid one folder being imported for multiple times
        uniqfds = list(set([f for orb in orbitals for f in orb.folders_]))
        # then index of the unique folders
        self.iuniqfds_ = [[uniqfds.index(f) for f in orb.folders_] for orb in self.orbitals_]

        # spillage coefficients
        OrbgenAssert(isinstance(spill_coefs, tuple) and len(spill_coefs) == 2,
                     'spill_coefs should be a tuple of two floats')
        self.spill_coefs_ = spill_coefs
        logging.info('') # spacing with the previous log
        logging.info(f'OrbgenCascade: spillage coefficients set to {self.spill_coefs_}')
        logging.info(f'OrbgenCascade: weight of S spillage (wov): {self.spill_coefs_[0]:.8e}')
        logging.info(f'OrbgenCascade: weight of T spillage (wop): {self.spill_coefs_[1]:.8e}')
        
        # Spillage type
        OrbgenAssert(mode in ['jy', 'pw'], 'mode should be either jy or pw')
        if mode == 'jy':
            self.minimizer_ = Spillage_jy() if optimizer.startswith('scipy') else SpillTorch_jy()
            for f in uniqfds:
                suffix = read_input_script(os.path.join(f, 'INPUT')).get('suffix', 'ABACUS')
                self.minimizer_.config_add(os.path.join(f, f'OUT.{suffix}'),
                                           weight=self.spill_coefs_)
        else:
            self.minimizer_ = Spillage_pw() if optimizer.startswith('scipy') else SpillTorch_pw()
            OLD_MATRIX_PW = dict([(f'orb_matrix_{i}', 
                                   f'orb_matrix.{i}.dat') 
                                  for i in range(2)])
            NEW_MATRIX_PW = dict([(f'orb_matrix_rcut{rcuts[0]}deriv{i}', 
                                   f'orb_matrix_rcut{rcuts[0]}deriv{i}.dat')
                                  for i in range(2)])
            for f in uniqfds:
                use_old = all([os.path.exists(os.path.join(f, v)) 
                               for v in OLD_MATRIX_PW.values()])
                use_new = all([os.path.exists(os.path.join(f, v)) 
                               for v in NEW_MATRIX_PW.values()])
                OrbgenAssert(use_old or use_new, 
                             f'{f} does not contain the matrix file')
                OrbgenAssert(not (use_old and use_new), 
                             f'{f} contains both the old and new matrix file')
                fmat = OLD_MATRIX_PW if use_old else NEW_MATRIX_PW
                fmat = {k: os.path.join(f, v) for k, v in fmat.items()}
                self.minimizer_.config_add(**fmat, weight=self.spill_coefs_)
        
        # constraint optimization settings
        self.fix_ = fix if fix else [[[] for _ in orb.nzeta_] for orb in orbitals]
        OrbgenAssert(len(self.fix_) == len(orbitals),
            '`fix_components` has inconsistent length with `orbitals`',
            RuntimeError) # possibly it is not the fault of user

        # summary
        logging.info(f'OrbgenCascade instantiated with {len(orbitals)} orbitals')
        
        # the flag for optimization, True means optimized, False means not
        # optimized, this is used to avoid re-optimizing the orbitals
        self.optimized_ = [False] * len(orbitals)
        # this is the internal status, so do not need to assert it

    def opt(self,
            diagnosis: bool = True,
            options: Optional[Dict] = None, 
            nthreads: int = 1,
            overwrite: bool = False) -> tuple['OrbgenCascade', List[float]]:
        '''optimize the cascade of orbitals
        
        Parameters
        ----------
        diagnosis : bool
            whether to diagnose the purity of the initial guess
        options : dict|None
            the options for the minimizer
        nthreads : int
            the number of threads to use
        overwrite : bool
            whether to overwrite the existing optimization results
        
        Returns
        -------
        OrbgenCascade
            the instance of the OrbgenCascade
        '''
        OrbgenAssert(self.initializer_ is not None,
                     'initializer should be set before optimization')
        
        nzmax = [np.array(orb.nzeta_) for orb in self.orbitals_]
        lmaxmax = np.max([len(nz) for nz in nzmax])
        nzmax = [np.pad(nz, (0, lmaxmax - len(nz)), 'constant') for nz in nzmax]
        nzmax = np.max(nzmax, axis = 0).tolist()

        spillage = []
        logging.info('') # spacing with the previous log
        logging.info(f'OrbgenCascade: start optimizing orbitals in cascade')
        for i, orb in enumerate(self.orbitals_):
            ichkpt, iconfs = self.chkptind_[i], self.iuniqfds_[i]
            orb_frozen, coefs_frozen, nzshift = None, None, None
            if ichkpt is not None:
                orb_frozen = self.orbitals_[ichkpt]
                coefs_frozen = [orb_frozen.coef_] # [it][l][n][q] -> float
                nzshift = orb_frozen.nzeta_
            
            if self.optimized_[i] and not overwrite:
                logging.info(f'OrbgenCascade: orbital {i} is already optimized, skip.')
                continue
            
            logging.info(f'OrbgenCascade: cascade calls for orbital {i} initialization')
            # only initialize when necessary
            if isinstance(self.minimizer_, (Spillage_jy, SpillTorch_jy)):
                # orb: OrbitalJY
                orb.init(nzshift=nzshift, 
                         diagnosis=diagnosis, 
                         **self.initializer_[i])
            else:
                # orb: OrbitalPW
                orb.init(nzmax=nzmax, 
                         nzshift=nzshift, 
                         diagnosis=diagnosis,
                         **self.initializer_[i]) # pw
            
            logging.info(f'OrbgenCascade: orbital {i} is initialized, start optimization')
            # then optimize
            # NOTE: the second param is always the inner, and the initial guess is always
            #       for the outer shell. The following function call will always only return
            #       the optimized outer shell (also with the optimized spillage value)
            coefs_shell, spillval = self.minimizer_.opt(
                coef_init=[orb.coef_], 
                coef_frozen=coefs_frozen,
                bounds=OrbgenCascade.bound([orb.coef_], coefs_frozen, self.fix_[i]),
                iconfs=iconfs, 
                ibands=orb.nbnds_,
                options=options, 
                nthreads=nthreads)
            orb.coef_ = merge(coefs_frozen, coefs_shell, 2)[0] if coefs_frozen else coefs_shell[0]
            spillage.append(spillval)
            self.optimized_[i] = True # mark this orbital as optimized
            # print
            logging.info(f'OrbgenCascade: orbital optimization ends with spillage = {spillval:.8e}')
            
        return self, spillage # fluent API
    
    def to_file(self, outdir) -> 'OrbgenCascade':
        '''plot the orbitals
        
        Parameters
        ----------
        outdir : str
            the folder to store the plot
        
        Returns
        -------
        OrbgenCascade
            the instance of the OrbgenCascade
        '''
        from SIAB.io.convention import orb as name_orb
        for orb in self.orbitals_:
            # name
            forb = orb.fn_ or name_orb(orb.elem_, orb.rcut_, orb.ecut_, orb.nzeta_)
            forb = os.path.join(outdir, forb)
            # save to file
            _ = orb.to_griddata(fn = forb, fpng = forb[:-4]+'.png')
            _ = orb.to_param(fn = forb[:-4]+'.param')
            
        return self
    
    def copy(self):
        '''
        copy the OrbgenCascade instance
        
        Returns
        -------
        OrbgenCascade
            the copy of the OrbgenCascade instance
        '''
        return deepcopy(self)
    
    def append(self, 
               orb: Orbital, 
               ichkpt: Optional[int]=None,
               fix: Optional[List[List[int]]]=None,
               initializer: Optional[Dict]=None):
        '''
        append one orbital to the cascade
        
        Parameters
        ----------
        orb : Orbital
            the orbital to append
        ichkpt : int|None
            the index of its inner shell orbital to be frozen, if None, will
            not freeze any inner shell orbital
        fix : list[list[int]]|None
            the indexes of zeta functions that are fixed, if None, will not
            fix any zeta functions
        initializer : dict|None
            the initializer for the orbital, if None, will use the default
            initializer
        
        Returns
        -------
        None
        '''        
        myfolders = set([f for myorb in self.orbitals_ for f in myorb.folders_])
        new_folders = [f for f in orb.folders_ if f not in myfolders]
        myfolders = list(myfolders.union(new_folders))
        
        if isinstance(self.minimizer_, (Spillage_jy, SpillTorch_jy)):
            for f in myfolders:
                suffix = read_input_script(os.path.join(f, 'INPUT')).get('suffix', 'ABACUS')
                self.minimizer_.config_add(os.path.join(f, f'OUT.{suffix}'),
                                           weight=self.spill_coefs_)
        elif isinstance(self.minimizer_, (Spillage_pw, SpillTorch_pw)):
            OLD_MATRIX_PW = dict([(f'orb_matrix_{i}', f'orb_matrix.{i}.dat')
                                  for i in range(2)]) # 0 for val, 1 for deriv
            NEW_MATRIX_PW = dict([(f'orb_matrix_rcut{orb.rcut_}deriv{i}', 
                                   f'orb_matrix_rcut{orb.rcut_}deriv{i}.dat')
                                 for i in range(2)]) # 0 for val, 1 for deriv
            for f in myfolders:
                use_old = all([os.path.exists(os.path.join(f, v)) 
                               for v in OLD_MATRIX_PW.values()])
                use_new = all([os.path.exists(os.path.join(f, v)) 
                               for v in NEW_MATRIX_PW.values()])
                OrbgenAssert(use_old or use_new, 
                             f'{f} does not contain the matrix file')
                OrbgenAssert(not (use_old and use_new), 
                             f'{f} contains both the old and new matrix file')
                fmat = OLD_MATRIX_PW if use_old else NEW_MATRIX_PW
                fmat = {k: os.path.join(f, v) for k, v in fmat.items()}
                self.minimizer_.config_add(**fmat, weight=self.spill_coefs_)
        
        # then append the orbital
        self.orbitals_.append(orb)
        self.chkptind_.append(ichkpt)
        self.iuniqfds_.append([myfolders.index(f) for f in orb.folders_])
        self.fix_.append(fix if fix is not None else [[] for _ in orb.nzeta_])
        self.initializer_.append(initializer or {'model': 'random'})
        self.optimized_.append(False)
    
    def get_num_orbitals(self) -> int:
        '''get the number of orbitals in the cascade
        
        Returns
        -------
        int
            the number of orbitals in the cascade
        '''
        return len(self.orbitals_)
    
    @staticmethod
    def bound(coef_init, 
              coef_frozen, 
              fix_components, 
              other=(-1, 1)):
        '''calculate the bound of the optimization
        
        Parameters
        ----------
        coef_init : list[list[list[list[float]]]]
            the initial guess of the coefficients to optimize, indexed by
            [it][l][n][q] -> float, type, angular momentum, zeta function,
            coefficient
        coef_frozen : list[list[list[list[float]]]]|None
            the coefficients of the frozen orbitals
        fix_components : list[list[int]
            the indexes of zeta functions that are fixed
        other : tuple
            for those not fixed, the bound of the coefficients

        Returns
        -------
        list[list[list[tuple]]]
            the bound of the coefficients only in coef_init
        '''
        from SIAB.spillage.listmanip import merge
        from SIAB.spillage.legacy.api import _coef_subset
        
        if fix_components is not None:
            logging.info('OrbgenCascade: `fix_component` is set, do constraint optimization on orbs.')
        
        fix_components = [[] for _ in coef_init[0]]\
            if fix_components is None else fix_components      
        coef = coef_init if coef_frozen is None else merge(coef_frozen, coef_init, 2)
        full = [[[(c, c) if iz in fix_components[l] else other for c in coef_lz]
                  for iz, coef_lz in enumerate(coef_l)]
                  for l, coef_l in enumerate(coef[0])]
        
        return [full] if coef_frozen is None else\
            _coef_subset(from_=full,
                         extract_=[len(b) for b in full],
                         exclude_=[len(c) for c in coef_frozen[0]])
'''
Concept
-------
orbgraph
    the graph expressing the relationship between orbitals. The initial one
    will always be an initializer (but can also leave as None, then will 
    use purely random number to initialize orbitals in each opt run), then
    with the connection (dependency) between orbitals, the optimizer can
    optimize the orbitals in a cascade manner.
'''
def build_orbgraph(elem,
                   rcut,
                   ecut,
                   primitive_type,
                   mode,
                   scheme, 
                   folders):
    '''build an orbgraph based on the scheme set by user.
    
    Parameters
    ----------
    elem : str
        the element of this cascade of orbitals
    rcut : float
        the cutoff radius of the orbitals
    ecut : float
        the kinetic energy cutoff of the underlying jy
    primitive_type : str
        the type of jy, can be `reduced` or `normalized`
    mode : str
        the mode of the optimization, can be `jy` or `pw`
    scheme : list[dict]
        the scheme of the orbitals, each element is a dict containing
        the information of the orbital, including nzeta, folders, nbnds
        and iorb_frozen, are number of zeta functions for each angular
        momentum, the folders where the orbital optimization will extract
        information, the number of bands to be included in the optimization
        and the index of its inner shell orbital to be frozen.
    folders : list[str]
        the folders where orbital optimization will extract information
    
    Returns
    -------
    dict: the orbgraph
    '''
    out = {'elem': elem, 'rcut': rcut, 'ecut': ecut, 'primitive_type': primitive_type, 
           'mode': mode, 'initializer': None, 'orbs': []}
    # this function is responsible for arranging orbitals so that there wont
    # be the case that one orbital refers to another that is not optimized
    # yet.
    if scheme.get('spill_guess') == 'atomic':
        _r = None if mode == 'pw' else rcut
        out['initializer'] = dft_folder(elem, 'monomer', None, _r)
    for orb in scheme['orbs']: # for each orbital...
        orb_ref = orb.get('checkpoint')
        orb_ref = orb_ref if orb_ref != 'none' or orb_ref is not None else None
        out['orbs'].append({
            'nzeta': orb['nzeta'],
            'nbnds': orb['nbands'],
            'iorb_frozen': orb_ref,
            'folders': [folders[i] for i in orb['geoms']]
        })
    return out

from SIAB.orb.orb_jy import OrbitalJY
class TestOrbgenCascade(unittest.TestCase):

    def setUp(self):
        here = os.path.dirname(os.path.abspath(__file__))
        self.testfiles = os.path.dirname(os.path.dirname(here))
        self.testfiles = os.path.join(self.testfiles, 'tests', 'integrated', 'Si_7au_jy')
        if not os.path.exists(self.testfiles):
            raise FileNotFoundError(f'Test files directory {self.testfiles} does not exist')
        
        self.orb_jy = [OrbitalJY(
            rcut=5.0, 
            ecut=50.0, 
            elem='Si', 
            nzeta=[i+1, i+1, i], 
            primitive_type='reduced', 
            folders=[os.path.join(self.testfiles, f'Si-dimer-{bl}-7au')
                     for bl in ['1.75', '2.00']],
            nbnds=[4, 4]
        ) for i in range(3)]
    
    def test_bound(self):
        
        coef_init = [[[0.1, 0.2], 
                      [0.3, 0.4]], 
                     [[0.5, 0.6], 
                      [0.7, 0.8]]] # [l][n][q] -> float
        coef_frozen = [[[0.9, 1.0], 
                        [1.1, 1.2]],
                       [[1.3, 1.4], 
                        [1.5, 1.6]]] # [l][n][q] -> float
        fix_components = [[0, 1, 2], [0, 1, 2, 3]]

        bound = OrbgenCascade.bound([coef_init], [coef_frozen], fix_components)
        self.assertEqual(bound, [[[[(0.1, 0.1), (0.2, 0.2)], 
                                   [(-1, 1), (-1, 1)]], 
                                  [[(0.5, 0.5), (0.6, 0.6)], 
                                   [(0.7, 0.7), (0.8, 0.8)]]]])

        bound = OrbgenCascade.bound([coef_init], None, fix_components)
        self.assertEqual(bound, [[[[(0.1, 0.1), (0.2, 0.2)], 
                                   [(0.3, 0.3), (0.4, 0.4)]], 
                                  [[(0.5, 0.5), (0.6, 0.6)], 
                                   [(0.7, 0.7), (0.8, 0.8)]]]])

    def test_orbgen_cascade_init(self):
        # test the instantiation of OrbgenCascade
        initializer = [{'model': 'atomic', 
                        'model_kwargs': {'jobdir': os.path.join(self.testfiles, 'Si-monomer-7au')}}
                        for _ in range(3)]
        cascade = OrbgenCascade(
            initializer=initializer,
            orbitals=self.orb_jy,
            chkpts=[None, 0, 1],
            mode='jy',
            optimizer='torch.swats',
            fix=[[[], []], [[], []], [[], []]]
        )
        self.assertEqual(len(cascade.orbitals_), 3)
        self.assertEqual(cascade.chkptind_, [None, 0, 1])
        self.assertTrue(all(set(i) == {0, 1} for i in cascade.iuniqfds_))
        self.assertEqual(cascade.fix_, [[[], []]] * 3)
        self.assertEqual(len(cascade.initializer_), 3)
        self.assertEqual(cascade.optimized_, [False] * 3)

    def test_orbgen_cascade_opt(self):
        initializer = [{'model': 'atomic', 
                        'model_kwargs': {'jobdir': os.path.join(self.testfiles, 'Si-monomer-7au')}}
                        for _ in range(3)]
        cascade = OrbgenCascade(
            initializer=initializer,
            orbitals=self.orb_jy,
            chkpts=[None, 0, 1],
            mode='jy',
            optimizer='scipy.bfgs'
        )
        cascade, spillage = cascade.opt()
        self.assertEqual(len(cascade.orbitals_), 3)
        self.assertTrue(all(cascade.optimized_))
        self.assertEqual(len(spillage), 3)
        self.assertTrue(all(isinstance(s, float) for s in spillage))
        self.assertEqual(len(cascade.initializer_), 3)

    def test_orbgen_cascade_append(self):
        initializer = [{'model': 'atomic', 
                        'model_kwargs': {'jobdir': os.path.join(self.testfiles, 'Si-monomer-7au')}}
                        for _ in range(3)]
        cascade = OrbgenCascade(
            initializer=initializer,
            orbitals=self.orb_jy,
            chkpts=[None, 0, 1],
            mode='jy',
            optimizer='scipy.bfgs'
        )
        new_orb = OrbitalJY(
            rcut=5.0, 
            ecut=50.0, 
            elem='Si', 
            nzeta=[3, 3, 2], 
            primitive_type='reduced', 
            folders=[os.path.join(self.testfiles, f'Si-dimer-{bl}-7au')
                     for bl in ['1.75', '2.00']],
            nbnds=[4, 4]
        )
        cascade.append(new_orb, 
                       ichkpt=2, 
                       fix=[[0, 1, 2], [0, 1, 2], [0, 1]])
        self.assertEqual(len(cascade.orbitals_), 4)
        self.assertEqual(cascade.chkptind_, [None, 0, 1, 2])
        self.assertTrue(all(set(i) == {0, 1} for i in cascade.iuniqfds_))
        self.assertEqual(cascade.fix_, [[[], [], []]] * 3 + [[[0, 1, 2], [0, 1, 2], [0, 1]]])
        self.assertDictEqual(cascade.initializer_[-1], {'model': 'random'})
        self.assertEqual(cascade.optimized_, [False] * 4)
        
    def test_orbgen_cascade_opt_append(self):
        initializer = [{'model': 'atomic', 
                        'model_kwargs': {'jobdir': os.path.join(self.testfiles, 'Si-monomer-7au')}}
                        for _ in range(3)]
        cascade = OrbgenCascade(
            initializer=initializer,
            orbitals=self.orb_jy,
            chkpts=[None, 0, 1],
            mode='jy',
            optimizer='scipy.bfgs'
        )
        cascade, _ = cascade.opt()
        
        new_orb = OrbitalJY(
            rcut=5.0, 
            ecut=50.0, 
            elem='Si', 
            nzeta=[3, 3, 2], 
            primitive_type='reduced', 
            folders=[os.path.join(self.testfiles, f'Si-dimer-{bl}-7au')
                     for bl in ['1.75', '2.00']],
            nbnds=[4, 4]
        )
        cascade.append(new_orb, 
                       ichkpt=2, 
                       fix=[[0, 1, 2], [0, 1, 2], [0, 1]])
        self.assertEqual(len(cascade.orbitals_), 4)
        self.assertEqual(cascade.chkptind_, [None, 0, 1, 2])
        self.assertTrue(all(set(i) == {0, 1} for i in cascade.iuniqfds_))
        self.assertEqual(cascade.fix_, [[[], [], []]] * 3 + [[[0, 1, 2], [0, 1, 2], [0, 1]]])
        self.assertDictEqual(cascade.initializer_[-1], {'model': 'random'})
        self.assertEqual(cascade.optimized_, [True] * 3 + [False])

    def test_orbgen_cascade_opt_opt(self):
        # test the restart of the optimization
        initializer = [{'model': 'atomic', 
                        'model_kwargs': {
                            'jobdir': os.path.join(self.testfiles, 'Si-monomer-7au')
                        }}
                        for _ in range(3)]
        cascade = OrbgenCascade(
            initializer=initializer,
            orbitals=self.orb_jy,
            chkpts=[None, 0, 1],
            mode='jy',
            optimizer='scipy.bfgs'
        )
        t = time.time()
        cascade, _ = cascade.opt()
        topt = time.time() - t
        
        # restart the optimization
        t = time.time()
        cascade, spillage = cascade.opt(overwrite=False)
        trestart = time.time() - t
        self.assertLess(trestart, topt) # should be fast since already optimized
        self.assertTrue(all(cascade.optimized_))
        self.assertEqual(len(spillage), 0)
        
    def test_orbgen_cascade_opt_append_opt(self):
        # test the optimization after appending an orbital
        initializer = [{'model': 'atomic', 
                        'model_kwargs': {
                            'jobdir': os.path.join(self.testfiles, 'Si-monomer-7au')
                        }}
                        for _ in range(3)]
        cascade = OrbgenCascade(
            initializer=initializer,
            orbitals=self.orb_jy,
            chkpts=[None, 0, 1],
            mode='jy',
            optimizer='scipy.bfgs'
        )
        t = time.time()
        cascade, spillage_1 = cascade.opt()
        tall = time.time() - t
        
        new_orb = OrbitalJY(
            rcut=5.0, 
            ecut=50.0, 
            elem='Si', 
            nzeta=[3, 3, 2], 
            primitive_type='reduced', 
            folders=[os.path.join(self.testfiles, f'Si-dimer-{bl}-7au')
                     for bl in ['1.75', '2.00']],
            nbnds=[4, 4]
        )
        cascade.append(new_orb, 
                       ichkpt=1, 
                       fix=[[], [], []],
                       initializer={'model': 'atomic', 
                                    'model_kwargs': {
                                        'jobdir': os.path.join(self.testfiles, 'Si-monomer-7au')
                                    }})
        self.assertListEqual(cascade.optimized_, [True, True, True, False])
        t = time.time()
        cascade, spillage_2 = cascade.opt(overwrite=False)
        t332 = time.time() - t
        self.assertLess(t332, tall) # because all components are fixed
        self.assertEqual(len(cascade.orbitals_), 4)
        self.assertTrue(all(cascade.optimized_))
        self.assertEqual(len(spillage_2), 1)
        self.assertEqual(spillage_2[0], spillage_1[-1]) # because it is the same!
        

if __name__ == '__main__':
    unittest.main()