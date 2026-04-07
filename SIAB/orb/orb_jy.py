'''
Brief
-----
the derived class of Orbital for handling specifically the jy as fit_basis case
'''
# in-built modules
import re
import os
import unittest
import logging

# third-party modules
import numpy as np

# local modules
from SIAB.orb.orb import Orbital
from SIAB.spillage.legacy.api import nzeta_infer
from SIAB.orb.jy_expmt import _coef_init
from SIAB.spillage.datparse import read_input_script
from SIAB.spillage.spillage import initgen_vloc
from SIAB.driver.control import OrbgenAssert

class OrbitalJY(Orbital):

    def __init__(self, 
                 rcut, 
                 ecut, 
                 elem, 
                 nzeta,
                 primitive_type,
                 folders,
                 nbnds):
        '''
        instantiate an orbital that can be optimized respect to calculation with
        jy basis set. This is corresponidng to the fit_basis = jy in the spillage
        json file. The nzeta will be inferred if it is set to 'auto'.
        
        Parameters
        ----------
        rcut: float
            the cutoff radius of the orbital, in Bohr
        ecut: float
            the kinetic energy cutoff of underlying jy basis that is used to 
            generate the NAO
        elem: str
            the element symbol
        nzeta: list[int]
            the number of zeta to use for each angular momentum.
        primitive_type: str
            the type of jy, can be `reduced` or `normalized`
        folders: list[str]
            the folders where the orbital optimization will extract
        nbnds: list[int]
            the number of bands to optimize for each folder
        '''
        super().__init__(rcut, ecut, elem, nzeta, primitive_type, folders, nbnds)
        if isinstance(self.nzeta_, str):
            m = re.match(r'^auto:(twsvd|amwsvd):(\d+(\.\d+)?)(:(max|mean))?$', self.nzeta_)
            OrbgenAssert(m is not None,
                         'the nzeta should be set to a list of integers or a string '
                         'with the format of auto:[method]:[threshold][:(max|mean)], '
                         'e.g. auto:twsvd:0.8:max')
            options = {'nbands': [nbnd.stop for nbnd in self.nbnds_],
                       'folders': self.folders_, 'statistics': m.group(5) or 'max',
                       'kernel': m.group(1), 'threshold': float(m.group(2))}
            self.nzeta_ = [int(np.ceil(nz)) for nz in nzeta_infer(**options)]
            logging.info(f'Summary: the nzeta is inferred as {self.nzeta_} '
                         f'using the {m.group(1)} method with a threshold of '
                         f'{m.group(2)} and the {m.group(5) or "max"} statistics.')

        logging.info('OrbitalJY instantiated.')

    def init(self, nzshift, diagnosis, **kwargs):
        '''
        initialize the orbital by assigning values to the contraction coefficients.
        
        Parameters
        ----------
        nzshift: List[int]
            
        '''
        nzshift = nzshift or [0] * len(self.nzeta_) # ensure nzshift is a list

        state_dict = kwargs.copy() | {'nzshift'  : nzshift, 
                                      'diagnosis': diagnosis}
        logging.info(f'OrbitalJY: initialize orbital with parameters:')
        for k, v in state_dict.items():
            logging.info(f'           {k}: {v}')
        del state_dict # avoid polluting the namespace

        # if the superclass can complete the initialization?
        coef = super().coefgen(nzshift=nzshift, 
                               diagnosis=diagnosis, 
                               **kwargs)
        # let's see
        if coef is not None: # the case 'random', 'ones'
            self.coef_ = coef
            logging.info('OrbitalJY: orbital initialized (coefs assigned).')
            return None

        # the implementation of model == 'atomic'
        jobdir = kwargs['model_kwargs']['jobdir']
        if 'OUT.' not in os.path.basename(jobdir):
            dftparam = read_input_script(os.path.join(jobdir, 'INPUT'))
            suffix = dftparam.get('suffix', 'ABACUS')
            jobdir = os.path.join(jobdir, f'OUT.{suffix}')
        # ATOMIC: the case requires the auxiliary potential to finish the `atomic` initialization
        fpsp     = kwargs['model_kwargs'].get('vloc_aux')
        lloc_min = kwargs['model_kwargs'].get('lloc_min', 4)
        self.coef_ = _coef_init(jobdir, self.nzeta_, nzshift, 
                                diagnosis=diagnosis,
                                lloc_min=lloc_min)
        
        # ATOMIC (Auxillary): the case requires the auxiliary potential
        if fpsp is not None and len(self.nzeta_) >= lloc_min + 1:
            # get the angular momenta for which the coefficients will be 
            # initialized with the auxiliary potential
            lloc = [i for i, _ in enumerate(self.nzeta_) if i >= lloc_min]
            # if there is no zeta, skip the initialization
            if all([nz == 0 for nz in self.nzeta_[lloc_min:]]):
                logging.info('OrbitalJY: orbital initialized (coefs assigned).')
                return None
            # solve the radial Schrodinger equation with given file which contains
            # the auxiliary potential, the angular momentum to solve, and the 
            # number of root to find
            temp = initgen_vloc(fpsp, lloc, self.nzeta_[lloc_min:], 
                                self.ecut_, self.rcut_, self.primitive_type_)
            logging.info('OrbitalJY: solve radial Schrodinger equation.')
            # only assign the coefficients for the user-defined angular momenta
            # and in the range of zeta functions from nzshift[l] to nzeta_[l]
            # the nzshift[l] is the starting index and nzeta_[l] is the end
            for l0, c_l0 in enumerate(temp):
                l = l0 + lloc_min
                self.coef_[l] = c_l0[nzshift[l]:self.nzeta_[l]]

        logging.info('OrbitalJY: orbital initialized (coefs assigned).')
        return None

class TestOrbitalJY(unittest.TestCase):
    def test_instantiate(self):
        here = os.path.dirname(__file__)
        # testfiles in another folder
        parent = os.path.dirname(here)
        outdir = os.path.join(parent, 'spillage/testfiles/Si/jy-7au/monomer-gamma/')

        orb = OrbitalJY(rcut=7, 
                        ecut=100, 
                        elem='Si', 
                        nzeta=[1, 1, 0], 
                        primitive_type='reduced', 
                        folders=[outdir], 
                        nbnds=[4])
        self.assertEqual(orb.nzeta_, [1, 1, 0])
    
    def test_init(self):
        # testfiles in another folder
        from os.path import dirname
        outdir = os.path.join(dirname(dirname(__file__)), 
            'spillage/testfiles/Si/jy-7au/monomer-gamma/')

        orb = OrbitalJY(rcut=7, 
                        ecut=100, 
                        elem='Si', 
                        nzeta=[1, 1, 0], 
                        primitive_type='reduced', 
                        folders=[outdir], 
                        nbnds=[4])
        suffix = read_input_script(os.path.join(outdir, 'INPUT')).get('suffix', 'ABACUS')
        orb.init(model='atomic',
                 model_kwargs={
                     'jobdir': os.path.join(outdir, f'OUT.{suffix}'),
                 },
                 nzshift=None, 
                 diagnosis=True)
        self.assertEqual(len(orb.coef_), 3) # up to l = 2


if __name__ == '__main__':
    unittest.main()
