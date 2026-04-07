'''
Concepts
--------
After calling DFT to calculate several quantities, the viewpoint
now fully transforms to the question "how to deal with an orbital".

On the other hand, 
the parameter list of Spillage.opt function defines a semi-mathematical
problem that:
def opt(self, coef_init, coef_frozen, iconfs, ibands,
        options, nthreads=1)
the first and the second parameters are purely initial guess and a fixed
component, which defines purely a mathematical problem of Spillage function
optimization. The lattering two are Spillage function specific, they
will be those terms building the Spillage function. The last two are
optimization options.

Classes defined here are for connecting between the (semi-)purely
mathematical problem and the physical (maybe?) problem, say the instance
of orbital.
'''
# in-built modules
import os
import unittest
import logging
from typing import Optional, List

# third-party modules
import numpy as np

# local modules
from SIAB.driver.control import OrbgenAssert
from SIAB.spillage.util import _spil_bnd_autoset
from SIAB.spillage.legacy.api import _coef_subset
from SIAB.spillage.radial import _nbes

# the `implemented_orbital_models` would be more organically used to organize
# the orbital initialization methods codes in the future
implemented_orbital_models = [
    'ones',       # jy
    'random',     # from the implementation of PHYSICAL REVIEW B103,235131(2021)
    'atomic',     # from atomic calculation
    'hydrogen',   # hydrogen-like orbitals, w/ and w/o slater screening
    'pretrained', # restart from an orbital
    ]
orbital_model_required_keys = {
    'ones': [],
    'random': ['seed'],
    'atomic': ['jobdir', 'vloc_aux', 'lloc_min'],
    'hydrogen': ['slater', 'otherelem'],
    'pretrained': ['pretrained']
}

def OrbitalAssert(rcut,
                  ecut,
                  elem,
                  nzeta,
                  primitive_type,
                  folders,
                  nbnds):
    OrbgenAssert(isinstance(rcut, (int, float)), 
                 'rcut should be a float or int',
                 TypeError)
    OrbgenAssert(isinstance(ecut, (int, float)),
                 'ecut should be a float or int',
                 TypeError)
    OrbgenAssert(isinstance(elem, str),
                 'elem should be a str',
                 TypeError)
    OrbgenAssert(isinstance(primitive_type, str),
                 'primitive_type should be a str',
                 TypeError)
    OrbgenAssert(primitive_type in ['reduced', 'normalized'],
                 'primitive_type should be either reduced or normalized',
                 ValueError)
    
    if folders is not None:
        OrbgenAssert(isinstance(folders, list) and \
                     all([isinstance(f, str) for f in folders]),
                     'folders should be a list of str',
                     TypeError)
        OrbgenAssert(all([os.path.exists(f) for f in folders]),
                     'some folders do not exist',
                     FileNotFoundError)

    if nbnds is not None:
        OrbgenAssert(len(nbnds) == len(folders),
                    'Mismatch of nbnds and folders when instantiating Orbital',
                    ValueError)

class Orbital:
    '''the orbital here corresponds to a set of coefficients of jy'''

    def __init__(self, 
                 rcut, 
                 ecut, 
                 elem, 
                 nzeta,
                 primitive_type,
                 folders=None,
                 nbnds=None):
        '''instantiate an orbital object, in this function the nzeta is
        calculated
        
        Parameters
        ----------
        rcut : float
            the cutoff radius of the orbital
        ecut : float
            the kinetic energy cutoff of the underlying jy
        elem : str
            the element of the orbital
        nzeta : list[int]|list[str]|None
            the number of zeta for each angular momentum
        primitive_type : str
            the type of jy, can be `reduced` or `normalized`. The latter
            is highly not recommended.
        folders : list[str]
            the folders where orbital optimization will extract information
        nbnds : list[range]|list[str]
            the number of bands to be included in the optimization. Besides
            the normal range, it can also be `occ` or `all`
        '''
        OrbitalAssert(rcut, ecut, elem, nzeta, primitive_type, folders, nbnds)
        self.rcut_ = rcut
        self.ecut_ = ecut
        self.elem_ = elem
        self.nzeta_ = nzeta # we will handle this in derived class in details
        self.primitive_type_ = primitive_type
        self.folders_ = folders
        self.nbnds_ = [range(_spil_bnd_autoset(nb, f)) 
                       for nb, f in zip(nbnds, folders)] \
                    if all([x is not None for x in [folders, nbnds]]) else nbnds
        self.coef_ = None

        # the customized filename
        self.fn_ = None

    def coefgen(self, 
                model: str='atomic',
                nzshift: Optional[List[int]]=None,
                diagnosis: Optional[bool]=None,
                **kwargs) -> Optional[List[List[np.ndarray]]]:
        '''the experimental refactored version of the init(), aiming to provide some
        model coefficients as the initial guess
        
        Parameters
        ----------
        method : str
            the method to initialize the orbital (coef), can be `'ones'`, `'random'`, 
            `'hydrogen'`, `'hydrogen-slater'` and `'pretrained'`. The `'atomic'` is also 
            allowed but will be performed in the derived class
        nzshift : Optional[List[int]]
            the shift of number of zeta for each angular momentum
        diagnosis : bool
            diagnose the purity of the initial guess
        
        There are also some additional keywords should be provided upon the value of
        initialization method. 
        - `'atomic'`:
            for the method that use the atomic calculation as the initial guess, the
            `jobdir` should be provided as the folder in which the atomic calculation
            is performed. Additionally, if high angular momentum orbitals like the
            g-orbital is required, empirically the `atomic` cannot give satisfying
            result, in this case, there are additional two keywords can be specified:
            `vloc_aux` and `lloc_min`. The former is for providing a local potential
            such that the orbital would be initialized by solving the radial
            Schrodinger equation in which the vloc, a local potential is employed, and
            the `lloc_min` determines beyond which angular momentum, will use this
            auxiliary local potential to perform the initial guess
        - `'pretrained'`:
            for the method that use a provided orbital file as the initial guess of 
            present orbital, the pretrained orbital file should be provided for the
            keyword `forb`
        '''
        # set default value for the diagnosis
        diagnosis = diagnosis or False
        OrbgenAssert(isinstance(diagnosis, bool), 
                     'diagnosis should be a bool',
                     TypeError)

        # dealing with the parameter `method`...
        method = 'atomic' if model is None else model.lower()
        OrbgenAssert(method in implemented_orbital_models, 
                     f'method {method} is not supported')
                
        if method == 'ones':
            return self._coefgen_ones()

        # more sophisticated method, the `model_kwargs` may be of need
        model_kwargs = kwargs.get('model_kwargs', {})
        if method == 'random':
            return self._coefgen_rand(
                nzshift, 
                random_seed=model_kwargs.get('random_seed'))

        if method == 'pretrained':
            return self._coefgen_file(
                nzshift, 
                forb=model_kwargs.get('pretrained'))
        
        if method == 'hydrogen':
            return self._coefgen_hydrogen(
                nzshift, 
                slater=model_kwargs.get('slater', False),
                otherelem=model_kwargs.get('otherelem'))

        if method == 'atomic':
            OrbgenAssert('jobdir' in model_kwargs, 
                        f'`jobdir` should be provided for `atomic` method: {kwargs}')
            return None # call for init() in derived class

        raise OrbgenAssert(False, f'method {method} is not supported')

    # FIXME: the name of subfunction to call now is inferred from the value of 
    #        the value of `method`. This is not a good design!
    def init(self, method=None, nzshift=None, diagnosis=True):
        '''calculate the initial value of the contraction coefficient of jy,
        crucial in further optimization tasks
        
        Parameters
        ----------
        method : 'random'|'ones'|str
            how to initialize the value of coeffs, can be either 'random', 'ones', or
            the directory where the single atomic calculation data is stored
        nzshift : list[int]|None
            the shift of number of zeta for each angular momentum
        diagnosis : bool
            diagnose the purity of the initial guess

        Returns
        -------
        List[List[np.ndarray]]
            the initial guess of primitive basis functions' coefficients
        '''
        OrbgenAssert(False, 'This function is deprecated, why you reached here?',
                     RuntimeError)

        # a brief sanity check
        OrbgenAssert(nzshift is None or isinstance(nzshift, list),
                     'nzshift should be a list of int or None',
                     TypeError)
        OrbgenAssert(isinstance(diagnosis, bool),
                     'diagnosis should be a bool',
                     TypeError)
        
        if method == 'random':
            return self._coefgen_rand(nzshift)
        elif method == 'ones':
            return self._coefgen_ones(nzshift)
        elif method == 'hydrogen': # use hydrogen-like orbitals
            logging.warning('Orbital: hydrogen method yields highly fluctuated orbital near '
                            'core, use with caution (high ecutjy). Also the numerical instability may occur.')
            return self._coefgen_hydrogen(nzshift)
        elif method == 'hydrogen-slater': # use hydrogen-like orbitals with slater screening
            logging.warning('Orbital: hydrogen method yields highly fluctuated orbital near '
                            'core, use with caution (high ecutjy). Also the numerical instability may occur.')
            return self._coefgen_hydrogen(nzshift, slater=True)
        elif os.path.isfile(method) and method.endswith('.orb'): # init from file
            return self._coefgen_file(nzshift, forb=method)
        elif not os.path.exists(method):
            OrbgenAssert(False, 
                         f'Initialize method set as `{method}`, while it does not exist',
                         FileNotFoundError)
        else: # the case spill_guess == 'atomic'
            logging.info(f'Initialize orbital with files in {method}')
        return None # derived class will call this function first, then if not None, return

    def _coefgen_rand(self, nzshift, random_seed: Optional[int] = None):
        '''generate a random set of coefficients'''
        logging.info('Orbital: initialize with random coefficients')
        less_dof = 0 if self.primitive_type_ == 'normalized' else 1

        # set the random seed, if provided
        if random_seed is not None:
            np.random.seed(random_seed)

        # generate
        coefs_rnd = [np.random.random((nz, _nbes(l, self.rcut_, self.ecut_) - less_dof)).tolist()
                        for l, nz in enumerate(self.nzeta_)]

        # get the subset of coefficients
        return _coef_subset(extract_=self.nzeta_, exclude_=nzshift, from_=coefs_rnd)[0] # [l][iz][q]

    def _coefgen_ones(self):
        less_dof = 0 if self.primitive_type_ == 'normalized' else 1
        return [np.eye(_nbes(l, self.rcut_, self.ecut_) - less_dof).tolist() 
                for l in range(len(self.nzeta_))] # [l][iz][q]
    
    def _coefgen_hydrogen(self, 
                          nzshift, 
                          slater=False,
                          otherelem: Optional[str] = None):
        '''
        Generate hydrogen-like orbitals.
        
        Parameters
        ----------
        nzshift : list[int]|None
            the shift of number of zeta for each angular momentum
        slater : bool
            whether to use slater screening
        otherelem : str|None
            the other element to use to generate the hydrogen-like orbitals, 
            if None, use the current element. This is useful when to avoid the
            truncation of the hydrogen-like orbital generated for present 
            element. Setting this value to another one with higher Z would
            work.

        Returns
        -------
        List[List[np.ndarray]]
            the initial guess of primitive basis functions' coefficients
        '''
        from SIAB.spillage.radial import proj_jl
        logging.info(f'Orbital: initialize with hydrogen-like orbitals (slater screening: {slater})')
        def __radgen(**kwargs): # rename with shorter name
            from SIAB.data.build import AtomSpecies
            return AtomSpecies.build_hydrogen_orb(**kwargs)
        
        elem = otherelem if otherelem is not None else self.elem_
        nzshift = [0] * len(self.nzeta_) if nzshift is None else nzshift
        return [[proj_jl(__radgen(elem=elem, 
                                  n=n+l+1, 
                                  l=l, 
                                  r=np.linspace(0, self.rcut_, int(self.rcut_/0.01)+1), 
                                  slater=slater),
                  dr=0.01, l=l, ecut=self.ecut_, rcut=self.rcut_, 
                  primitive_type=self.primitive_type_)[0].tolist()
                 for n in range(nlo, nhi)]
                for l, (nlo, nhi) in enumerate(zip(nzshift, self.nzeta_))]
    
    def _coefgen_file(self, nzshift, forb):
        from SIAB.spillage.spillage import initgen_file
        logging.info(f'Orbital: generate coefficients from file: {forb}')
        temp = {'forb': forb, 'nzeta': self.nzeta_, 'ecut': self.ecut_,
                'rcut': self.rcut_, 'dr': 0.01, 'primitive_type': self.primitive_type_}
        return _coef_subset(extract_=self.nzeta_, 
                            exclude_=nzshift, 
                            from_=initgen_file(**temp))[0]

    def coef(self):
        '''return the contraction coefficient of jy'''
        OrbgenAssert(self.coef_ is not None, 
                     'coef not initialized', 
                     ValueError)
        return self.coef_
    
    def __eq__(self, value):
        '''compare two orbitals'''
        if not isinstance(value, Orbital):
            return False
        return self.rcut_ == value.rcut_ and\
               self.ecut_ == value.ecut_ and\
               self.elem_ == value.elem_ and\
               self.nzeta_ == value.nzeta_ and\
               self.primitive_type_ == value.primitive_type_ and\
               self.folders_ == value.folders_ and\
               self.nbnds_ == value.nbnds_ and\
               self.coef_ == value.coef_

    def __ne__(self, value):
        '''compare two orbitals'''
        return not self.__eq__(value)

    def to_griddata(self, r=None, dr=0.01, fn=None, fpng=None):
        '''
        Map the orbital to real-space grid
        
        Parameters
        ----------
        r : list[float]|None
            the real-space grid, if None, use uniform grid with dr
        dr : float
            the grid spacing, default is 0.01 in bohr
        fn : str|None
            the file name to store the orbital in real-space grid. if None,
            no file will be stored
        fpng : str|None
            the file name to store the plot of the orbital in real-space grid.
            if None, no plot will be stored
        
        Returns
        -------
        list[list[np.ndarray]]: the orbital on real-space grid, indexed by 
        [l][iz][q][r] -> float
        '''
        # third-party modules
        import matplotlib.pyplot as plt
        
        # local modules
        from SIAB.spillage.radial import build_reduced, build_raw, coeff_normalized2raw
        from SIAB.spillage.orbio import write_nao
        from SIAB.spillage.plot import plot_chi
        
        # real-space grid
        r = np.linspace(0, self.rcut_, int(self.rcut_ / dr) + 1) if r is None else r
        # convert    
        chi = build_reduced(self.coef_, self.rcut_, r, True) \
            if self.primitive_type_ in ['reduced', 'nullspace', 'svd'] \
            else build_raw(coeff_normalized2raw(self.coef_, self.rcut_),
                           self.rcut_, r, 0.0, True)
        # additional, if write
        if any(f is not None for f in [fn, self.fn_]):
            fn = fn or self.fn_
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            write_nao(fpath=fn, 
                      elem=self.elem_,
                      ecut=self.ecut_,
                      rcut=self.rcut_,
                      nr=len(r),
                      dr=dr,
                      chi=chi)
        # additional, if plot
        if fpng:
            os.makedirs(os.path.dirname(fpng), exist_ok=True)
            plot_chi(chi=chi, r=r, save=fpng)
            plt.close()
        # return
        return chi

    def to_param(self, sigma=0.0, fn=None):
        '''
        convert the orbital to those coefficients of raw jy
        
        Parameters
        ----------
        sigma : float
            the sigma of the Gaussian, if None, use the default value
        fn : str|None
            the file name to store the orbital in real-space grid. if None,
            no file will be stored
        
        Returns
        -------
        list[list[np.ndarray]]: the coefficients of raw jy, indexed by
        [l][iz][q] -> float
        '''
        from SIAB.spillage.radial import coeff_normalized2raw, coeff_reduced2raw
        from SIAB.spillage.orbio import write_param
        coeff2raw = coeff_reduced2raw \
            if self.primitive_type_ in ['reduced', 'nullspace', 'svd'] \
            else coeff_normalized2raw
        coeff_raw = coeff2raw(self.coef_, self.rcut_)
        if fn:
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            write_param(fpath=fn,
                        coeff=coeff_raw,
                        rcut=self.rcut_,
                        sigma=sigma,
                        elem=self.elem_,)
        return coeff_raw
    
    @staticmethod
    def nloc(nzeta):
        '''calculate the number of dimensions of the local orbital'''
        return np.sum([(2*l+1)*nz for l, nz in enumerate(nzeta)])
    
class TestOrbital(unittest.TestCase):
    
    here = os.path.dirname(__file__)
    # testfiles in another folder
    parent = os.path.dirname(here)
    outdir = os.path.join(parent, 'spillage/testfiles/Si/jy-7au/monomer-gamma/')
    
    def test_instantiate(self):
        orb = Orbital(rcut=7, 
                      ecut=100, 
                      elem='Si', 
                      nzeta=[1, 1, 0], 
                      primitive_type='reduced', 
                      folders=[self.outdir], 
                      nbnds=[4])
        self.assertEqual(orb.nzeta_, [1, 1, 0])
        self.assertEqual(orb.rcut_, 7)
        self.assertEqual(orb.ecut_, 100)
        self.assertEqual(orb.elem_, 'Si')
        self.assertEqual(orb.primitive_type_, 'reduced')
        
    def test_coefgen_random_noshift(self):
        orb = Orbital(rcut=7, 
                      ecut=100, 
                      elem='Si', 
                      nzeta=[1, 1, 0], 
                      primitive_type='reduced', 
                      folders=[self.outdir], 
                      nbnds=[4])
        out = orb.coefgen('random')
        for o, nz in zip(out, [1, 1, 0]):
            self.assertEqual(len(o), nz)
        # however, it is not set to the self.coef_, so it is still undefined
        self.assertEqual(orb.coef_, None)

    def test_coefgen_random_shift(self):
        orb = Orbital(rcut=7, 
                      ecut=100, 
                      elem='Si', 
                      nzeta=[1, 1, 0], 
                      primitive_type='reduced', 
                      folders=[self.outdir], 
                      nbnds=[4])
        out = orb.coefgen('random', [1, 1, 0], True)
        for o in out:
            self.assertEqual(len(o), 0)

    def test_coefgen_ones(self):
        orb = Orbital(rcut=7, 
                      ecut=100, 
                      elem='Si', 
                      nzeta=[0, 0, 0], 
                      primitive_type='reduced', 
                      folders=[self.outdir], 
                      nbnds=[4])
        out = orb.coefgen('ones') # [l][iz][q]
        self.assertEqual(len(out), 3)
        self.assertTrue(all([np.allclose(np.array(o), np.eye(len(o[0]))) for o in out]))
        

    def test_coefgen_hydrogen(self):
        orb = Orbital(rcut=7, 
                      ecut=100, 
                      elem='Si', 
                      nzeta=[1, 1, 0], 
                      primitive_type='reduced', 
                      folders=[self.outdir], 
                      nbnds=[4])
        out = orb.coefgen('hydrogen')
        self.assertTrue(isinstance(out[0][0][0], float)) # should be correctly indexed by [l][iz][q]

if __name__ == "__main__":
    unittest.main()
