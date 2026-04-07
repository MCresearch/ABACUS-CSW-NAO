'''
this file contains interfaces and tool functions for calling functions to optimize
the Spillage function. The interface in principle needs the following information:
- elem: the element symbol
- rcut(s): the cutoff radius for the orbitals, outside of which the orbitals are tru-
    ncated
- ecut: the kinetic energy cutoff for spherical wave functions
- fit_basis: jy or pw
- optimizer: none, restart or bfgs
- maxiter: the maximum number of iterations
- nthreads: the number of threads used in optimization
- folders of ABACUS runs
- orbgen parameters, for each orb, there are: 
    - nzeta: number of zeta functions for each angular momentum l
    - frozen: for hierarchical optimization strategy, a reference to previously
        optimized orbitals. If valid, will only optimize its own layer of zeta
        functions
    - spillage reference: defines how practically the spillage function of this
        orbital is calculated. For each folder, what stores in it should be the
        calculation on one specific geometry (geom) under one specific 
        deformation/perturbation (pert). For each calculation, it is possible
        to select the band range to ref. Thus most primitively, what should be
        provided is a list of tuple (str, range), in which the str is the folder
        name and the range is the band range to ref. If the range is not provided,
        the whole band range will be used.
'''
# in-built modules
import os
import re
import unittest
import logging

# third-party modules
import numpy as np
import matplotlib.pyplot as plt

# local modules
from SIAB.driver.control import OrbgenAssert
from SIAB.spillage.spillage import Spillage_jy, initgen_jy
from SIAB.spillage.datparse import read_wfc_lcao_txt, read_triu, \
    read_running_scf_log, read_input_script, read_istate_info
from SIAB.spillage.lcao_wfc_analysis import api as wfc_analysis
from SIAB.spillage.util import literal_eval as leval

def _coef_subset(from_, extract_, exclude_=None):
    """
    from coefs indexed by [l][z][q], get the nzeta `get_` from `from_` and
    substract those in `substract_`

    Parameters
    ----------
    from_: list[list[list[float]]]
        usually the coefficients of the orbitals, indexed like [l][z][q] -> float
    extract_: list[int]
        the number of zeta functions for each l
    exclude_: list[int]
        the number of zeta functions for each l, default is None

    
    Returns
    -------
    list[list[list[float]]]: the subset
    """
    if exclude_ is None:
        return [[[ from_[l][iz] for iz in range(nz) ] for l, nz in enumerate(extract_)]]
    OrbgenAssert(len(extract_) >= len(exclude_), 
        f"(at least) size error of extract_ ({len(extract_)}) vs. exclude_ ({len(exclude_)})")
    # zero padding...
    exclude_ = exclude_ + (len(extract_) - len(exclude_))*[0]
    OrbgenAssert(all([nz >= nz0 for nz, nz0 in zip(extract_, exclude_)]),
        f"not hirarcal structure of these two sets: exclude_={exclude_} while extract_={extract_}")
    iz_subset = [list(range(iz, jz)) for iz, jz in zip(exclude_, extract_)]
    # then get coefs from data with iz_subset, the first dim of iz_subset is l
    #               l  zeta                 l  list of zeta
    return [[[ from_[l][j] for j in jz ] for l, jz in enumerate(iz_subset)]]

def nzeta_infer(nbands, 
                folders, 
                statistics = 'max',
                kernel = 'twsvd',
                threshold = 1.0):
    """infer the nzeta from given folders with some strategy. If there are
    multiple kpoints calculated, for each folder the result will be firstly
    averaged and used to represent the `nzeta` inferred from the whole folder
    , then the `nzeta` will be averaged again over all folders to get the
    final result
    
    Parameters
    ----------
    nbands: int|list[int]
        the number of bands for each folder. If it is a list, the range of
        bands will be set individually for each folder.
    folders: list[str]
        the folders where the ABACUS run information are stored.
    statistics: str
        the statistics method used to infer nzeta, can be 'max' or 'mean'
    kernel: str
        the population analysis kernel used to infer nzeta, can be 'twsvd'
        or 'amwsvd'. For detailed explanation, please see the function
        `_nzeta_infer_core`. Default is 'twsvd'.
    threshold: float|list[float]|list[list[int]]|None
        can be either float or list of floats, which means the threshold for 
        the significance for each atomtype. 
        If it is a list[list[int]], it should be the nzeta for each atomtype, 
        then the loss will be evaluated based on it. If it is None, will be 
        overwritten to the list[float] case in which all elements are 1.0

    Returns
    -------
    nzeta: list[int]
        the inferred nzeta for each folder
    """
    OrbgenAssert(statistics in ['max', 'mean'],
                 f'ERROR: statistics method {statistics} is not supported')
    OrbgenAssert(isinstance(folders, list), 
                 f"folders should be a list: {folders}")
    OrbgenAssert(all([isinstance(f, str) for f in folders]), 
                 f"folders should be a list of strings: {folders}")
    
    logging.info('nzeta-inference task triggered')
    logging.info(f'statistics: {statistics} (max or mean, to summarize over structures)')
    temp = 'twsvd or amwsvd, Typewise-Wavefunction-Singular-Value-Decomposition or '
    temp += 'Atomwise-Maximum-Wavefunction-Singular-Value-Decomposition'
    logging.info(f'kernel: {kernel} ({temp})')
    temp = 'float, to determine the available information for optimizing the zeta'
    temp += ' functions.'
    logging.info(f'threshold: {threshold} ({temp})')
    logging.info('')

    # nzeta = np.array([0])
    nzeta = []
    nbands = [nbands] * len(folders) if not isinstance(nbands, list) else nbands

    max_shape = (0,) # 1D array
    for folder, nband in zip(folders, nbands):
        nzeta_ = np.array(_nzeta_infer_core(folder, nband, kernel, threshold=threshold))
        max_shape = np.maximum(max_shape, nzeta_.shape)
        nzeta.append(nzeta_)

    nzeta = np.array([nz.reshape(max_shape).tolist() for nz in nzeta])
    return {'max': np.max, 'mean': np.mean}[statistics](nzeta, axis=0).tolist()

def _nzeta_infer_core(folder, nband, kernel = 'twsvd', threshold = 1.0):
    """infer nzeta based on one structure whose calculation result is stored
    in the folder
    
    Parameters
    ----------
    folder: str
        the folder where the ABACUS run information are stored
    nband: int|list[int]|range
        if specified as int, it is the highest band index to be considered. 
        if specified as list or range, it is the list of band indexes to be
        considered
    kernel: str, optional
        the population analysis method used to infer nzeta, can be 'twsvd'
        or 'amwsvd', means Typewise-Wavefunction-Singular-Value-Decomposition
        or Atomwise-Maximum-Wavefunction-Singular-Value-Decomposition. For
        more information, please see the function implementation in file
        SIAB/spillage/lcao_wfc_analysis.py, functions
        `atomwise_maximum_wavefunction_singular_value_decomposition` and 
        `typewise_wavefunction_singular_value_decomposition`. default is 'twsvd'.
    threshold: float|list[float]|list[list[int]]|None
        can be either float or list of floats, which means the threshold for 
        the significance for each atomtype. 
        If it is a list[list[int]], it should be the nzeta for each atomtype, 
        then the loss will be evaluated based on it. If it is None, will be 
        overwritten to the list[float] case in which all elements are 1.0

    Returns
    -------
    np.ndarray: the inferred nzeta for the folder, like list[int]
    """

    # read INPUT and running_*.log
    params = read_input_script(os.path.join(folder, "INPUT"))
    outdir = os.path.abspath(os.path.join(folder, 
        "OUT." + params.get("suffix", "ABACUS")))
    nspin = int(params.get("nspin", 1))
    fwfc = "WFC_NAO_GAMMA" if params.get("gamma_only", False) else "WFC_NAO_K"
    running = read_running_scf_log(os.path.join(outdir, 
        f"running_{params.get('calculation', 'scf')}.log"))
    kpts, ener, occ = read_istate_info(os.path.join(outdir, "istate.info"))

    OrbgenAssert(nspin == running["nspin"],
        f"nspin in INPUT and running_scf.log are different: {nspin} and {running['nspin']}")
    
    # if nspin == 2, the "spin-up" kpoints will be listed first, then "spin-down"
    wk = running["wk"]

    logging.info(f'Perform nzeta_infer/evaluation task for {folder}')
    nzeta = np.array([0])
    for isk in range(nspin*len(wk)): # loop over (ispin, ik)
        ik, is_ = isk % len(wk), isk % nspin
        w = wk[ik] # spin-up and spin-down share the wk
        wfc, _, _, kpt = read_wfc_lcao_txt(os.path.join(outdir, f"{fwfc}{isk+1}.txt"))

        OrbgenAssert(isinstance(nband, (int, str)),
            f'ERROR: nband should be int or str, but got {nband}')
        nband = nband if isinstance(nband, int) else leval('n' + nband)
        OrbgenAssert(wfc.shape[1] >= nband, \
            f"ERROR: number of bands for orbgen is larger than calculated: {nband} > {wfc.shape[1]}",
            ValueError)

        # the complete return list is (wfc.T, e, occ, k)
        ovlp = read_triu(os.path.join(outdir, f"data-{isk}-S"))
        # the number of non-zeros of each l is the maximal number of zeta functions
        sigma, nz, loss = wfc_analysis(wfc, ovlp, running['natom'], running['nzeta'], 
                                       method=kernel, nband=nband, threshold=threshold)
        # print out the result
        logging.info(f'k = {ik}, ispin = {is_}')
        for it, (st_, nzt_) in enumerate(zip(sigma, nz)):
            logging.info(f'For type {it}, do SVD on {nband} band(s), '\
                  'the complete list of sigma values are:')
            for l, s in enumerate(st_):
                logging.info(f'l = {l}:')
                temp = ''
                for i, s_ in enumerate(s):
                    temp += f'{s_:>8.4f} '
                    if i % 5 == 4:
                        logging.info(temp)
                        temp = ''
                logging.info('')
            logging.info(f'Number of zeta functions for type {it} is {nzt_}')
        logging.info(f'jy space truncation (subspace) loss: {loss:8.4e}\n')
        logging.info('')
        
        nz = np.array(nz[0])
        nzeta = np.resize(nzeta, np.maximum(nzeta.shape, nz.shape)) + nz * w / nspin

    # count the number of atoms
    OrbgenAssert(len(running["natom"]) == 1, 
                 f"multiple atom types are not supported: {running['natom']}",
                 NotImplementedError)

    return nzeta

class TestAPI(unittest.TestCase):

    # @unittest.skip('Skip for developement')
    def test_nzeta_to_initgen(self):

        nz1 = np.random.randint(0, 5, 2).tolist()
        nz2 = np.random.randint(0, 5, 3).tolist()
        nz3 = np.random.randint(0, 5, 4).tolist()
        nz4 = np.random.randint(0, 5, 5).tolist()
        lmax = max([len(nz) for nz in [nz1, nz2, nz3, nz4]]) - 1
        total_init = [(lambda nzeta: nzeta + (lmax + 1 - len(nzeta))*[-1])(nz) for nz in [nz1, nz2, nz3, nz4]]
        total_init = [max([orb[i] for orb in total_init]) for i in range(lmax + 1)]
        for iz in range(lmax + 1):
            self.assertEqual(total_init[iz], max([
                nz[iz] if iz < len(nz) else -1 for nz in [nz1, nz2, nz3, nz4]]))

    # @unittest.skip('Skip for developement')
    def test_coefs_subset(self):

        nz3 = [3, 3, 2]
        nz2 = [2, 2, 1]
        nz1 = [1, 1]
        data = [np.random.random(i).tolist() for i in nz3]
        
        subset = _coef_subset(extract_=nz1, 
                              exclude_=None, 
                              from_=data)
        self.assertEqual(subset, [[[data[0][0]], 
                                   [data[1][0]]]])

        subset = _coef_subset(extract_=nz2, 
                              exclude_=None, 
                              from_=data)
        self.assertEqual(subset, [[[data[0][0], data[0][1]], 
                                   [data[1][0], data[1][1]],
                                   [data[2][0]]
                                  ]])

        subset = _coef_subset(extract_=nz3, 
                              exclude_=None, 
                              from_=data)
        self.assertEqual(subset, [[[data[0][0], data[0][1], data[0][2]], 
                                   [data[1][0], data[1][1], data[1][2]],
                                   [data[2][0], data[2][1]]
                                  ]])

        subset = _coef_subset(extract_=nz3, 
                              exclude_=nz2, 
                              from_=data)
        self.assertEqual(subset, [[[data[0][2]], 
                                   [data[1][2]], 
                                   [data[2][1]]]])
        
        subset = _coef_subset(extract_=nz2, 
                              exclude_=nz1, 
                              from_=data)
        self.assertEqual(subset, [[[data[0][1]], 
                                   [data[1][1]], 
                                   [data[2][0]]]])
        
        subset = _coef_subset(extract_=nz3, 
                              exclude_=nz1, 
                              from_=data)
        self.assertEqual(subset, [[[data[0][1], data[0][2]], 
                                   [data[1][1], data[1][2]], 
                                   [data[2][0], data[2][1]]]])
        
        nz2 = [1, 2, 1]
        subset = _coef_subset(extract_=nz3, 
                              exclude_=nz2, 
                              from_=data)
        self.assertEqual(subset, [[[data[0][1], data[0][2]], 
                                   [data[1][2]], 
                                   [data[2][1]]]])

        subset = _coef_subset(extract_=nz3, 
                              exclude_=nz3, 
                              from_=data)
        self.assertEqual(subset, [[[], [], []]])
        
        with self.assertRaises(ValueError):
            subset = _coef_subset(extract_=nz1, 
                                  exclude_=nz3, 
                                  from_=data) # nz1 < nz3

        with self.assertRaises(ValueError):
            subset = _coef_subset(extract_=nz2, 
                                  exclude_=nz3, 
                                  from_=data) # nz2 < nz3

        subset = _coef_subset(extract_=[2, 2, 0], 
                              exclude_=[2, 1, 0], 
                              from_=data)
        self.assertEqual(subset, [[[],
                                   [data[1][1]],
                                   []]])

    @unittest.skip('still under development')
    def test_nzeta_infer_core(self):

        here = os.path.dirname(__file__)
        # gamma case is easy, multi-k case is more difficult
        fpath = os.path.join(here, "testfiles/Si/jy-7au/monomer-gamma/")

        nzeta = _nzeta_infer_core(fpath, 4, 'amwsvd')
        ref = [1, 1, 0]
        self.assertTrue(all([abs(nz - ref[i]) < 1e-8 for i, nz in enumerate(nzeta)]))
        nzeta = _nzeta_infer_core(fpath, 5, 'amwsvd')
        ref = [2, 1, 0]
        self.assertTrue(all([abs(nz - ref[i]) < 1e-8 for i, nz in enumerate(nzeta)]))
        nzeta = _nzeta_infer_core(fpath, 10, 'amwsvd')
        ref = [2, 1, 1]
        self.assertTrue(all([abs(nz - ref[i]) < 1e-8 for i, nz in enumerate(nzeta)]))

        # multi-k case
        fpath = os.path.join(here, "testfiles/Si/jy-7au/monomer-k/")
        testref = """
  1.000  0.000  0.000   
  0.000  1.000  0.000   
  0.000  1.000  0.000   
  0.000  1.000  0.000   
  1.000  0.000  0.000   
  0.000  0.000  1.000   
  0.000  0.000  1.000   
  0.000  0.000  1.000   
  0.000  0.000  1.000   
   0.000  0.000  1.000
  0.999  0.000  0.001   
  0.004  0.991  0.005   
  0.000  1.000  0.000   
  0.000  1.000  0.000   
  0.719  0.004  0.277   
 -0.000 -0.000  1.000   
  0.129  0.427  0.444   
  0.000  0.008  0.992   
  0.000  0.008  0.992   
   0.000  0.000  1.000
  0.999  0.001  0.001   
  0.007  0.989  0.004   
 -0.000  0.991  0.009   
  0.000  0.999  0.001   
  0.392  0.005  0.603   
  0.014  0.079  0.907   
 -0.000  0.357  0.643   
  0.310  0.336  0.354   
  0.000  0.012  0.988   
   0.000  0.000  1.000
  0.999  0.001 -0.000   
  0.011  0.987  0.002   
  0.000  0.990  0.010   
  0.000  0.990  0.010   
  0.049  0.116  0.836   
  0.000  0.031  0.969   
  0.000  0.031  0.969   
  0.000  0.286  0.714   
  0.000  0.286  0.714   
   0.547  0.303  0.150
"""
        refdata = [list(map(float, line.split())) for line in testref.strip().split("\n")]
        refdata = np.array(refdata).reshape(4, -1, 3) # reshape to (nks, nbands, lmax+1)
        wk = [0.0370, 0.2222, 0.4444, 0.2963]
        degen = np.array([2*i + 1 for i in range(3)], dtype=float)

    #@unittest.skip('still under development')
    def test_nzeta_infer_core_occ(self):
        here = os.path.dirname(__file__)
        # gamma case is easy, multi-k case is more difficult
        fpath = os.path.join(here, "../testfiles/Si/jy-7au/monomer-gamma/")
        with self.assertRaises(ValueError):
            nzeta = _nzeta_infer_core(fpath, 4, 'wll')
        nzeta = _nzeta_infer_core(fpath, 4)
        logging.info(nzeta)

    @unittest.skip('still under development')
    def test_nzeta_mean_conf(self):

        here = os.path.dirname(__file__)

        # first test the monomer case at gamma
        fpath = os.path.join(here, "testfiles/Si/jy-7au/monomer-gamma/")
        folders = [fpath]
        """Here I post the accumulated wll matrix from unittest testtypewise_wavefunction_angular_momentum_decomposition_gamma
        at SIAB/spillage/lcao_wfc_analysis.py:61
        
        Band 1     1.000  0.000  0.000     sum =  1.000
        Band 2     0.000  1.000  0.000     sum =  1.000
        Band 3     0.000  1.000  0.000     sum =  1.000
        Band 4     0.000  1.000  0.000     sum =  1.000
        Band 5     1.000  0.000  0.000     sum =  1.000
        Band 6     0.000  0.000  1.000     sum =  1.000
        Band 7     0.000  0.000  1.000     sum =  1.000
        Band 8     0.000  0.000  1.000     sum =  1.000
        Band 9     0.000  0.000  1.000     sum =  1.000
        Band 10     0.000  0.000  1.000     sum =  1.000
        """
        # nbands = 4, should yield 1s1p as [1, 1, 0]
        nzeta = nzeta_infer(4, folders, 'mean')
        nzeta = [int(np.round(nz)) for nz in nzeta] # filter to integer
        self.assertEqual(nzeta, [1, 1, 0])
        # nbands = 5, should yield 2s1p as [2, 1, 0]
        nzeta = nzeta_infer(5, folders)
        nzeta = [int(np.round(nz)) for nz in nzeta] # filter to integer
        self.assertEqual(nzeta, [2, 1, 0])
        # nbands = 10, should yield 2s1p1d as [2, 1, 1]
        nzeta = nzeta_infer(10, folders)
        nzeta = [int(np.round(nz)) for nz in nzeta] # filter to integer
        self.assertEqual(nzeta, [2, 1, 1])

        # then test the multi-k case
        fpath = os.path.join(here, "testfiles/Si/jy-7au/monomer-k/")
        folders = [fpath]
        """Here I post the accumulated wll matrix from unittest testtypewise_wavefunction_angular_momentum_decomposition_multi_k
        at SIAB/spillage/lcao_wfc_analysis.py:87
        
        ik = 0, wk = 0.0370
        Band 1     1.000  0.000  0.000     sum =  1.000
        Band 2     0.000  1.000  0.000     sum =  1.000
        Band 3     0.000  1.000  0.000     sum =  1.000
        Band 4     0.000  1.000  0.000     sum =  1.000
        Band 5     1.000  0.000  0.000     sum =  1.000
        Band 6     0.000  0.000  1.000     sum =  1.000
        Band 7     0.000  0.000  1.000     sum =  1.000
        Band 8     0.000  0.000  1.000     sum =  1.000
        Band 9     0.000  0.000  1.000     sum =  1.000
        Band 10     0.000  0.000  1.000     sum =  1.000
        ik = 1, wk = 0.2222
        Band 1     0.999  0.000  0.001     sum =  1.000
        Band 2     0.004  0.991  0.005     sum =  1.000
        Band 3     0.000  1.000  0.000     sum =  1.000
        Band 4     0.000  1.000  0.000     sum =  1.000
        Band 5     0.719  0.004  0.277     sum =  1.000
        Band 6    -0.000 -0.000  1.000     sum =  1.000
        Band 7     0.129  0.427  0.444     sum =  1.000
        Band 8     0.000  0.008  0.992     sum =  1.000
        Band 9     0.000  0.008  0.992     sum =  1.000
        Band 10     0.000  0.000  1.000     sum =  1.000
        ik = 2, wk = 0.4444
        Band 1     0.999  0.001  0.001     sum =  1.000
        Band 2     0.007  0.989  0.004     sum =  1.000
        Band 3    -0.000  0.991  0.009     sum =  1.000
        Band 4     0.000  0.999  0.001     sum =  1.000
        Band 5     0.392  0.005  0.603     sum =  1.000
        Band 6     0.014  0.079  0.907     sum =  1.000
        Band 7    -0.000  0.357  0.643     sum =  1.000
        Band 8     0.310  0.336  0.354     sum =  1.000
        Band 9     0.000  0.012  0.988     sum =  1.000
        Band 10     0.000  0.000  1.000     sum =  1.000
        ik = 3, wk = 0.2963
        Band 1     0.999  0.001 -0.000     sum =  1.000
        Band 2     0.011  0.987  0.002     sum =  1.000
        Band 3     0.000  0.990  0.010     sum =  1.000
        Band 4     0.000  0.990  0.010     sum =  1.000
        Band 5     0.049  0.116  0.836     sum =  1.000
        Band 6     0.000  0.031  0.969     sum =  1.000
        Band 7     0.000  0.031  0.969     sum =  1.000
        Band 8     0.000  0.286  0.714     sum =  1.000
        Band 9     0.000  0.286  0.714     sum =  1.000
        Band 10     0.547  0.303  0.150     sum =  1.000
        """
        # nbands = 4, should yield 
        #   [1/1, 3/3, 0]*0.0370 
        # + [1/1, 3/3, 0]*0.2222 
        # + [1/1, 3/3, 0]*0.4444 
        # + [1/1, 3/3, 0]*0.2963
        # = [1, 1, 0]
        nzeta = nzeta_infer(4, folders, 'mean', 'amwsvd')
        nzeta = [int(np.round(nz)) for nz in nzeta] # filter to integer
        self.assertEqual(nzeta, [1, 1, 0])
        # nbands = 5, should yield
        #   [2/1 + 0,     3/3 + 0,       0]*0.0370 
        # + [1/1 + 0.719, 3/3 + 0,       0.283/5]*0.2222 
        # + [1/1 + 0.392, 3/3 + 0,       0.618/5]*0.4444 
        # + [1/1 + 0.049, 3/3 + 0.116/3, 0.858/5]*0.2963
        # = [1, 1, 0]
        nzeta = nzeta_infer(5, folders, 'mean', 'wll')
        nzeta = [int(np.round(nz)) for nz in nzeta] # filter to integer
        self.assertEqual(nzeta, [1, 1, 0])
        # nbands = 10, should yield
        #   [2/1, 3/3, 5/5] * 0.0370
        # + [1.848, 3.438/3, 4.711/5] * 0.2222
        # + [1.722, 3.769/3, 4.51/5] * 0.4444
        # + [1.607, 4.021/3, 4.37/5] * 0.2963
        # = [1.726, 1.247, 0.906] = [2, 1, 1]
        nzeta = nzeta_infer(10, folders, 'mean', 'wll')
        nzeta = [int(np.round(nz)) for nz in nzeta] # filter to integer
        self.assertEqual(nzeta, [2, 1, 1])

        # test the two folder mixed case, monomer-gamma and monomer-k
        fpath1 = os.path.join(here, "testfiles/Si/jy-7au/monomer-gamma/")
        fpath2 = os.path.join(here, "testfiles/Si/jy-7au/monomer-k/")
        folders = [fpath1, fpath2]
        # nbands = 4, should yield [1, 1, 0]
        nzeta = nzeta_infer(4, folders, 'mean', 'wll')
        nzeta = [int(np.round(nz)) for nz in nzeta] # filter to integer
        self.assertEqual(nzeta, [1, 1, 0])
        # nbands = 5, should yield [2, 1, 0]
        nzeta = nzeta_infer(5, folders, 'mean', 'wll')
        nzeta = [int(np.round(nz)) for nz in nzeta] # filter to integer
        self.assertEqual(nzeta, [2, 1, 0])
        # nbands = 10, should yield [2, 1, 1]
        nzeta = nzeta_infer(10, folders, 'mean', 'wll')
        nzeta = [int(np.round(nz)) for nz in nzeta] # filter to integer
        self.assertEqual(nzeta, [2, 1, 1])

        # test the dimer-1.8-gamma
        fpath_dimer = os.path.join(here, "testfiles/Si/jy-7au/dimer-1.8-gamma/")
        folders = [fpath_dimer]
        nzeta_dimer_nbnd4 = nzeta_infer(4, folders, 'mean', 'wll')
        nzeta_dimer_nbnd5 = nzeta_infer(5, folders, 'mean', 'wll')
        nzeta_dimer_nbnd10 = nzeta_infer(10, folders, 'mean', 'wll')
        # also get the monomer-gamma result
        fpath_mono = os.path.join(here, "testfiles/Si/jy-7au/monomer-gamma/")
        folders = [fpath_mono]
        nzeta_mono_nbnd4 = nzeta_infer(4, folders, 'mean', 'wll')
        nzeta_mono_nbnd5 = nzeta_infer(5, folders, 'mean', 'wll')
        nzeta_mono_nbnd10 = nzeta_infer(10, folders, 'mean', 'wll')
        # the mixed case should return average of the two
        nzeta_mixed_nbnd4 = nzeta_infer(4, [fpath_dimer, fpath_mono], 'mean', 'wll')
        self.assertEqual(nzeta_mixed_nbnd4, 
                         [(a+b)/2 for a, b in\
                          zip(nzeta_dimer_nbnd4, nzeta_mono_nbnd4)])
        nzeta_mixed_nbnd5 = nzeta_infer(5, [fpath_dimer, fpath_mono], 'mean', 'wll')
        self.assertEqual(nzeta_mixed_nbnd5, 
                         [(a+b)/2 for a, b in\
                          zip(nzeta_dimer_nbnd5, nzeta_mono_nbnd5)])
        nzeta_mixed_nbnd10 = nzeta_infer(10, [fpath_dimer, fpath_mono], 'mean', 'wll')
        self.assertEqual(nzeta_mixed_nbnd10, 
                         [(a+b)/2 for a, b in\
                          zip(nzeta_dimer_nbnd10, nzeta_mono_nbnd10)])

    @unittest.skip('This is not a unittest. Instead, this \
is a minimal example to investigate the synergetic \
effect of the two parameters, nzeta and nband on the \
orbital generation task.')
    def test_sigma_nzeta_nbands(self):
        '''for doing numerical experiments, according to sigma value,
        determine the nzeta that can produce the best orbital genreation
        results. Possible adjustable parameters are mainly in two 
        aspects:
        - nband
        - nzeta
        
        that is, the range of bands to be considered
        and the number of zeta to be generated.

        The initial guess also matters but it is clear.
        '''
        from SIAB.spillage.datparse import read_istate_info
        from SIAB.spillage.lcao_wfc_analysis import typewise_wavefunction_singular_value_decomposition

        rcut = 6
        nzeta = [2, 2, 0]
        # Al: 2s 2p valence electrons
        # 1s2, 2s2, 2p6, 3s2, 3p1
        ibands_atom = [0, 4, 1, 2, 3, 5, 6, 7]
        jobdir = '/root/documents/simulation/orbgen/Test1Aluminum-20241011'
        outdir = [f'Al-dimer-2.00-{rcut}au',
                  f'Al-dimer-2.50-{rcut}au',
                  f'Al-dimer-3.00-{rcut}au',
                  f'Al-dimer-3.75-{rcut}au',
                  f'Al-dimer-4.50-{rcut}au']
        occ_thr = 1e-1 # threshold on occ to determine nbands

        nthreads = 4
        option = {"maxiter": 2000, "disp": False, "ftol": 0, 
                  "gtol": 1e-6, 'maxcor': 20}

        #######
        # Run #
        #######
        minimizer = Spillage_jy()
        nbands = [0] * len(outdir)
        for i, f in enumerate(outdir):
            suffix = f
            d = os.path.join(jobdir, f, f'OUT.{suffix}')
            _, _, occ = read_istate_info(os.path.join(d, 'istate.info'))
            logging.info(f'For {f}, occ:')
            for isp, occ_sp in enumerate(occ):
                logging.info(f'spin = {isp}')
                for ik, occ_k in enumerate(occ_sp):
                    logging.info(f'k = {ik}')
                    for io, o in enumerate(occ_k):
                        logging.info(f'{o:.4e}', end=' ')
                        if io % 5 == 4:
                            logging.info('')
                    logging.info('')
                logging.info('')
            logging.info('')
            nbands[i] = len(np.where(np.array(occ) > occ_thr)[0])
            minimizer.config_add(d)

        # svd
        for i, f in enumerate(outdir):
            nbnd = nbands[i]
            d = os.path.join(jobdir, f, f'OUT.{f}')
            wfc = read_wfc_lcao_txt(os.path.join(d, "WFC_NAO_GAMMA1.txt"))[0]
            ovlp = read_triu(os.path.join(d, "data-0-S"))
            running = read_running_scf_log(os.path.join(d, "running_scf.log"))
            sigma = typewise_wavefunction_singular_value_decomposition(wfc, ovlp, nbnd, running["natom"],
                                running["nzeta"], 1.0)
            logging.info(f'For {f}, nbnd = {nbnd}, sigma:')
            for l, s in enumerate(sigma[0]):
                logging.info(f'l = {l}')
                for i, si in enumerate(s):
                    logging.info(f'{si:.4e}', end=' ')
                    if i % 5 == 4:
                        logging.info('')
                logging.info('')
            logging.info('')
        ibands = [range(n) for n in nbands]

        ###########################################################
        # the following should be a new way to give initial guess #
        ###########################################################
        suffix = f'Al-monomer-{rcut}au'
        ibands_atom = [[[0], [4]], 
                       [[1, 2, 3], [5, 6, 7]]]
        
        coef_init = [[] for _ in nzeta] # without type dimension
        for l, nz in enumerate(nzeta):
            for iz in range(nz):
                ib = ibands_atom[l][iz]
                nz_ = np.zeros_like(np.array(nzeta))
                nz_[l] = 1
                c = initgen_jy(os.path.join(jobdir, suffix, f'OUT.{suffix}'),
                                       nz_,
                                       ibands=ib,
                                       diagnosis=True)
                coef_init[l].append(c[l][0])

if __name__ == "__main__":
    unittest.main()
