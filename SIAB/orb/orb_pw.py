'''
Brief
-----
derived class of Orbital for handling specifically the pw as fit_basis case
'''
# in-built modules
import os
import unittest
import logging

# local modules
from SIAB.driver.control import OrbgenAssert
from SIAB.orb.orb import Orbital
from SIAB.orb.cascade import OrbgenCascade
from SIAB.spillage.spillage import initgen_pw
from SIAB.spillage.legacy.api import _coef_subset

from SIAB.spillage.listmanip import nestpat

class OrbitalPW(Orbital):
    '''the derived class of Orbital for handling specifically the pw
    as fit_basis case'''

    def __init__(self, 
                 rcut, 
                 ecut, 
                 elem, 
                 nzeta,
                 primitive_type,
                 folders,
                 nbnds):
        '''Initialize the `OrbitalPW` class for handling the plane wave (pw) fit basis.

        Note:
            For `OrbitalPW`, the `nzeta` inference from `nbnds` is not supported.
         '''
        super().__init__(rcut, ecut, elem, nzeta, primitive_type, folders, nbnds)
        # for OrbitalPW, the nzeta-infer is not supported
        logging.info('OrbitalPW instantiated.')
        
    def init(self, nzmax, nzshift, diagnosis = True, **kwargs):
        '''Initialize the contraction coefficients for the plane wave (pw) basis.
        This function differs from other `Orbital` derived classes by extracting
        the full set of initial coefficient guesses and then selecting a subset.
        
        Parameters
        ----------
        srcdir: str
            the directory containing the matrix file. Unlike OrbitalJY, the orb_matrix
            file will be in the jobdir, instead of outdir
        nzmax: list[int]
            the maximum number of zeta for each angular momentum
        nzshift: list[int]
            the starting index from which initial guess of coefs will be extracted
        diagnosis: bool
            whether to print the diagnostic information
        '''
        from SIAB.spillage.spillage import initgen_vloc
        state_dict = kwargs.copy() | {'nzmax'    : nzmax, 
                                      'nzshift'  : nzshift, 
                                      'diagnosis': diagnosis}
        logging.info(f'OrbitalPW: initialize orbital with parameters:')
        for k, v in state_dict.items():
            logging.info(f'           {k}: {v}')
        del state_dict # avoid polluting the namespace

        # if the superclass can complete the initialization?
        coef = super().coefgen(nzshift=nzshift, 
                               diagnosis=diagnosis, 
                               **kwargs)
        # let's see
        if coef is not None:
            self.coef_ = coef
            logging.info('OrbitalPW: orbital initialized (coefs assigned).')
            return None

        # the implementation of model == 'atomic'
        OLD_MATRIX = 'orb_matrix.0.dat'
        NEW_MATRIX = f'orb_matrix_rcut{self.rcut_}deriv0.dat'

        jobdir = kwargs['model_kwargs']['jobdir']
        OrbgenAssert(os.path.exists(jobdir), 
                     f'{jobdir} does not exist',
                     FileNotFoundError)
        # any one of the matrix file should exist
        OrbgenAssert(any([os.path.exists(os.path.join(jobdir, f))\
                          for f in [OLD_MATRIX, NEW_MATRIX]]),
                     f'{jobdir} does not contain the matrix file',
                     FileNotFoundError)
        # but not both
        OrbgenAssert(not all([os.path.exists(os.path.join(jobdir, f))\
                              for f in [OLD_MATRIX, NEW_MATRIX]]),
                     f'{jobdir} contains both the old and new matrix file',
                     FileExistsError)
        
        # ATOMIC: the case requires the auxiliary potential to finish the `atomic` initialization
        fmat = OLD_MATRIX if os.path.exists(os.path.join(jobdir, OLD_MATRIX)) \
            else NEW_MATRIX
        coef_full = initgen_pw(os.path.join(jobdir, fmat), nzmax) # coef[l][zeta][q] -> float
        self.coef_ = _coef_subset(extract_=self.nzeta_, 
                                  exclude_=nzshift, 
                                  from_=coef_full)[0]
        fpsp     = kwargs['model_kwargs'].get('vloc_aux')
        lloc_min = kwargs['model_kwargs'].get('lloc_min', 4)

        # ATOMIC (Auxillary): the case requires the auxiliary potential
        if fpsp is not None and len(self.nzeta_) >= lloc_min + 1:
            # get the angular momenta for which the coefficients will be 
            # initialized with the auxiliary potential
            lloc = [i for i, _ in enumerate(self.nzeta_) if i >= lloc_min]
            # if there is no zeta, skip the initialization
            if all([nz == 0 for nz in self.nzeta_[lloc_min:]]):
                logging.info('OrbitalPW: orbital initialized (coefs assigned).')
                return None
            # solve the radial Schrodinger equation with given file which contains
            # the auxiliary potential, the angular momentum to solve, and the 
            # number of root to find
            temp = initgen_vloc(fpsp, lloc, self.nzeta_[lloc_min:], 
                                self.ecut_, self.rcut_, self.primitive_type_) # temp[l][zeta][q] -> float
            logging.info('OrbitalPW: solve radial Schrodinger equation.')
            # zero-padding the coefficients to match the nbes[l]
            nbes = [len(c[0]) if c else 0 for c in coef_full] # the dim for each l
            temp = [[clz + [0]*(nbes[l] - len(clz)) for clz in cl] for l, cl in enumerate(temp)]
            # only assign the coefficients for the user-defined angular momenta
            # and in the range of zeta functions from nzshift[l] to nzeta_[l]
            # the nzshift[l] is the starting index and nzeta_[l] is the end
            for l0, c_l0 in enumerate(temp):
                l = l0 + lloc_min
                self.coef_[l] = c_l0[nzshift[l]:self.nzeta_[l]]
        
        logging.info('OrbitalPW: orbital initialized (coefs assigned).')

class TestOrbitalPW(unittest.TestCase):

    def test_instantiate(self):
        here = os.path.dirname(__file__)
        # testfiles in another folder
        parent = os.path.dirname(here)
        outdir = os.path.join(parent, 'spillage/testfiles/Si/pw/monomer-gamma/')

        orb = OrbitalPW(rcut=7, 
                        ecut=100, 
                        elem='Si', 
                        nzeta=[1, 1, 0], 
                        primitive_type='reduced', 
                        folders=[outdir], 
                        nbnds=[4])
        self.assertEqual(orb.nzeta_, [1, 1, 0])

    def test_opt_initrnd(self):
        here = os.path.dirname(__file__)
        # testfiles in another folder
        parent = os.path.dirname(here)
        outdir = os.path.join(parent, 'spillage/testfiles/Si/pw/monomer-gamma/')

        orb = OrbitalPW(rcut=7, 
                        ecut=40, 
                        elem='Si', 
                        nzeta=[1, 1, 0], 
                        primitive_type='reduced', 
                        folders=[outdir], 
                        nbnds=[4])
        options = {"maxiter": 10, "disp": False, "ftol": 0, "gtol": 1e-6, 'maxcor': 20}

        with self.assertRaises(NotImplementedError):
            cascade = OrbgenCascade([{'model': 'random'}],
                                    [orb],
                                    [None],
                                    mode = 'pw')
            # forbs = cascade.opt(diagnosis=True, options=options)
            # self.assertEqual(len(forbs), 0) # no orbital will be saved

    def test_opt_initatomic(self):
        here = os.path.dirname(__file__)
        # testfiles in another folder
        parent = os.path.dirname(here)
        outdir = os.path.join(parent, 'spillage/testfiles/Si/pw/monomer-gamma/')

        orb = OrbitalPW(rcut=7, 
                        ecut=40, 
                        elem='Si', 
                        nzeta=[1, 1, 0], 
                        primitive_type='reduced', 
                        folders=[outdir], 
                        nbnds=[4])
        options = {"maxiter": 10, "disp": False, "ftol": 0, "gtol": 1e-6, 'maxcor': 20}

        with self.assertRaises(NotImplementedError):
            cascade = OrbgenCascade([{'model': 'atomic', 
                                      'model_kwargs': {
                                          'jobdir': outdir
                                      }}],
                                    [orb],
                                    [None],
                                    mode = 'pw')
            # forbs = cascade.opt(diagnosis=True, options=options)
            # self.assertEqual(len(forbs), 0) # no orbital will be saved

if __name__ == "__main__":
    unittest.main()
