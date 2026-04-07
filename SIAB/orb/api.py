# in-built modules
import os
import re
import logging
import unittest
from typing import List, Dict

# local modules
from SIAB.driver.control import OrbgenAssert
from SIAB.orb.orb_jy import OrbitalJY
from SIAB.orb.orb_pw import OrbitalPW
from SIAB.orb.orb import orbital_model_required_keys
from SIAB.orb.cascade import OrbgenCascade

def OrbAPIAssert(elem,
                 rcut,
                 ecut,
                 nzeta,
                 mode,
                 primitive_type,
                 folders,
                 nbnds,
                 iorb_frozen,
                 optimizer):
    '''check the input of GetOrbCascadeInstance'''
    OrbgenAssert(isinstance(elem, str), 
                 'elem should be a str')
    OrbgenAssert(isinstance(rcut, (int, float)), 
                 'rcut should be a float or int')
    OrbgenAssert(isinstance(ecut, (int, float)),
                 'ecut should be a float or int')
    OrbgenAssert(isinstance(nzeta, list),
                 'nzeta should be a list')
    
    if mode == 'pw':
        cond = all([isinstance(nz_it, list) for nz_it in nzeta]) or\
               all([isinstance(nz_it, str) for nz_it in nzeta])
        OrbgenAssert(cond, 'nzeta should be a list of list or str')
    else:
        cond = all([isinstance(nz_it, (list, str)) for nz_it in nzeta]) or\
               all([isinstance(nz, int) for nz_it in nzeta for nz in nz_it if isinstance(nz_it, list)]) or\
               all([re.match(r'^auto:(twsvd|amwsvd):(\d+(\.\d+)?)(:(max|mean))?$', nz) 
                    for nz_it in nzeta for nz in nz_it if isinstance(nz_it, str)])
        OrbgenAssert(cond, 'nzeta should be a list of list[int] or a special string')

    OrbgenAssert(isinstance(primitive_type, str),
                 'primitive_type should be a str',
                 TypeError)        
    OrbgenAssert(primitive_type in ['reduced', 'normalized'],
                 'primitive_type should be either reduced or normalized',
                 ValueError)
    
    cond = isinstance(folders, list) or\
           all([isinstance(fd_it, list) for fd_it in folders]) or\
           all([isinstance(fd, str) for fd_it in folders for fd in fd_it])
    OrbgenAssert(cond, 'folders should be a list of list of str')
    OrbgenAssert(all([os.path.exists(fd) for fd_it in folders for fd in fd_it]),
                 'some folders do not exist',
                 FileNotFoundError)

    cond = isinstance(nbnds, list) or\
           all([isinstance(nbnd_it, list) for nbnd_it in nbnds]) or\
           all([isinstance(nbnd, (range, str, int)) for nbnd_it in nbnds for nbnd in nbnd_it])
    OrbgenAssert(cond, 'nbnds should be a list of list of int, range or str')    

    cond = isinstance(iorb_frozen, list) and\
           all([isinstance(orb_i, int) or orb_i is None for orb_i in iorb_frozen])
    OrbgenAssert(cond, 'iorb_frozen should be a list of int or None')

    OrbgenAssert(isinstance(mode, str), 'mode should be a str')    
    OrbgenAssert(mode in ['jy', 'pw'], 'mode should be either jy or pw')

    OrbgenAssert(isinstance(optimizer, str), 'optimizer should be a str')
    OrbgenAssert(re.match(r'^torch\..*|scipy\..*', optimizer),
                 'currently only optimizer implemented under torch and scipy are'
                 f' supported, got {optimizer}')

def GetOrbCascadeInstance(elem,
                          rcut, 
                          ecut, 
                          primitive_type,
                          initializer, 
                          orbparam,
                          mode,
                          optimizer='torch.swats',
                          **kwargs) -> OrbgenCascade:
    '''build an instance/task for optimizing orbitals in a cascade
    
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
    orbgraph : list[dict]
        the graph of the orbitals, each element is a dict containing
        the information of the orbital, including nzeta, folders, nbnds
        and iorb_frozen, are number of zeta functions for each angular
        momentum, the folders where the orbital optimization will extract
        information, the number of bands to be included in the optimization
        and the index of its inner shell orbital to be frozen.
    mode : str
        the mode of the optimization, can be `jy` or `pw`

    Returns
    -------
    OrbgenCascade
        the instance of the OrbgenCascade
    '''
    OrbgenAssert(orbparam is not None, 'orbparam should not be None')

    # distribute the parameter for each orbital
    nzeta, folders, chkpts = tuple(map(list, zip(
        *[[o.get(k) for k in ['nzeta', 'folders', 'iorb_frozen']]
          for o in orbparam])))
    autoset = lambda x, n: x if isinstance(x, list) else [x]*n
    nbnds = [autoset(orb.get('nbnds'), len(folders[i])) 
             for i, orb in enumerate(orbparam)]
    
    # Changelog at 2025-12-25 (Merry Christmas!)
    # now it is allowed to set the initializer for each orbital
    # initiailizer passed to this function is a dict like 
    # {'model': 'ones'} / {'model': 'atomic', 'jobdir': 'path/to/jobdir'} / ...

    # Changelog at 2025-12-29
    # change the interface to `model` and `model_kwargs` to make the orbital
    # initialization more flexible. Now the initializer passed to this function
    # is a dict like {'model': 'atomic', 'model_kwargs': {'jobdir': 'path/to/jobdir'}}

    # first prepare the default parameter for each orbital
    initializer = [initializer.copy() for _ in range(len(orbparam))]
    # then overwrite if the orbital has its own initialization parameters
    for i, (custom, orbinit) in enumerate(zip(orbparam, initializer)):
        model = custom.get('model', orbinit['model'])
        if model != orbinit['model']:
            logging.info(f'Orbital {i}: initialization model overwritten with \'{model}\'')
            orbinit['model'] = model
        # update the model_kwargs if needed
        custom_kwargs = custom.get('model_kwargs', {})
        orbinit.setdefault('model_kwargs', {}).update(custom_kwargs)
        # remove the keys that are not required by the present model
        orbinit['model_kwargs'] = {k: v for k, v in orbinit['model_kwargs'].items() 
                                   if k in orbital_model_required_keys[model]}

    # do the check
    OrbAPIAssert(elem, rcut, ecut, nzeta, mode, primitive_type, 
                 folders, nbnds, chkpts, optimizer)
    
    # describe the orbital
    GetOrbitalInstance = {'jy': OrbitalJY, 'pw': OrbitalPW}[mode]
    myorbs = [GetOrbitalInstance(rcut=rcut,                     # cutoff radius of orbital to gen
                                 ecut=ecut,                     # kinetic energy cutoff
                                 elem=elem,                     # a element symbol
                                 nzeta=nz,                      # number of zeta functions of each l
                                 primitive_type=primitive_type, # `reduce` or `normalized`
                                 folders=fd,                    # folders in which reference data is stored
                                 nbnds=nbnd)                    # number of bands to refer
            for nz, fd, nbnd in zip(nzeta, folders, nbnds)]
    
    # Changelog at 2025-12-26
    # find if there is any customized filename settings for orbitals
    for orb, orbp in zip(myorbs, orbparam):
        orb.fn_ = orbp.get('filename') # or None :)
    
    # plug orbitals in optimization cascade
    # the spillage should be also a parameter of the cascade, instead of a in-built member
    return OrbgenCascade(orbitals=myorbs, 
                         # orbital optimization parameters
                         initializer=initializer, 
                         chkpts=chkpts, 
                         optimizer=optimizer,
                         fix=[p.get('fix_components') for p in orbparam],
                         # spillage parameters
                         # FIXME: the orbgencacade should not build spillage the mathematical
                         #        object by itself, instead, it should be passed to the
                         #        cascade.
                         mode=mode, 
                         spill_coefs=tuple(kwargs.get('__iop_spill_coefs__', (0, 1.0))))

def DeriveCascadeInstance(elem,
                          rcut,
                          ecut,
                          primitive_type,
                          cascade: OrbgenCascade, 
                          orbparam: List[Dict]) -> OrbgenCascade:
    '''
    derive a new OrbgenCascade instance from an existing one,
    with newly added orbitals.
    
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
    cascade : OrbgenCascade
        the existing OrbgenCascade instance
    orbparam : list[dict]
        the parameters of the new orbitals to be added,
        each element is a dict containing the information of the orbital,
        including nzeta, folders, nbnds and iorb_frozen, are number of zeta
        functions for each angular momentum, the folders where the orbital
        optimization will extract information, the number of bands to be
        included in the optimization and the index of its inner shell orbital
        to be frozen.
        
    Returns
    -------
    OrbgenCascade
        the new OrbgenCascade instance with the new orbitals added
    '''
    OrbgenAssert(isinstance(cascade, OrbgenCascade), 
                 'cascade should be an instance of OrbgenCascade')
    OrbgenAssert(orbparam is not None, 'orbparam should not be None')
    
    nzeta = [orb.get('nzeta') for orb in orbparam]
    folders = [orb.get('folders') for orb in orbparam]
    autoset = lambda x, n: x if isinstance(x, list) else [x]*n
    nbnds = [autoset(orb.get('nbnds'), len(folders[i])) for i, orb in enumerate(orbparam)]
    
    GetOrbitalInstance = None
    if all([isinstance(orb, OrbitalJY) for orb in cascade.orbitals_]):
        GetOrbitalInstance = OrbitalJY
    elif all([isinstance(orb, OrbitalPW) for orb in cascade.orbitals_]):
        GetOrbitalInstance = OrbitalPW
    else:
        raise TypeError('Ill-defined OrbgenCascade instance provided: '
                        'all orbitals within the cascade should be of the same type')
    orbs = [GetOrbitalInstance(rcut=rcut,                     # cutoff radius of orbital to gen
                               ecut=ecut,                     # kinetic energy cutoff
                               elem=elem,                     # a element symbol
                               nzeta=nz,                      # number of zeta functions of each l
                               primitive_type=primitive_type, # `reduce` or `normalized`
                               folders=fd,                    # folders in which reference data is stored
                               nbnds=nbnd)                    # number of bands to refer
            for nz, fd, nbnd in zip(nzeta, folders, nbnds)]
    
    cascade_new = cascade.copy()
    initializer = cascade_new.initializer_[-1]
    for i, orb in enumerate(orbs):
        cascade_new.append(orb, 
                           ichkpt=orbparam[i].get('iorb_frozen'),
                           fix=orbparam[i].get('fix_components'),
                           initializer=initializer | orbparam[i])
    
    return cascade_new

class TestOrbAPI(unittest.TestCase):
    
    def setUp(self):
        here = os.path.dirname(os.path.abspath(__file__))
        self.testfiles = os.path.dirname(os.path.dirname(here))
        self.testfiles = os.path.join(self.testfiles, 'tests', 'integrated', 'Si_7au_jy')
        if not os.path.exists(self.testfiles):
            raise FileNotFoundError(f'Test files directory {self.testfiles} does not exist')
    
    def test_get_orb_cascade_instance(self):
        '''test the GetOrbCascadeInstance function'''
        elem = 'Si'
        rcut = 5.0
        ecut = 20.0
        primitive_type = 'reduced'
        initializer = None
        orbparam = [{'nzeta': [2, 2, 2], 
                     'folders': [os.path.join(self.testfiles, f'Si-dimer-{bl}-7au')
                                 for bl in ['1.75', '2.00']], 
                     'nbnds': [8, 8], 
                     'iorb_frozen': None}] # only one orbital to generate
        mode = 'jy'
        
        cascade = GetOrbCascadeInstance(elem, rcut, ecut, primitive_type, 
                                        initializer, orbparam, mode)
        self.assertIsInstance(cascade, OrbgenCascade)
        self.assertEqual(len(cascade.orbitals_), 1)
        self.assertEqual(cascade.optimized_, [False])

    def test_derive_cascade_instance(self):
        '''test the DeriveCascadeInstance function'''
        elem = 'Si'
        rcut = 5.0
        ecut = 20.0
        primitive_type = 'reduced'
        initializer = None
        orbparam = [{'nzeta': [1, 1, 0], 
                     'folders': [os.path.join(self.testfiles, f'Si-dimer-{bl}-7au')
                                 for bl in ['1.75', '2.00']], 
                     'nbnds': [8, 8], 
                     'iorb_frozen': None}]
        mode = 'jy'
        cascade = GetOrbCascadeInstance(elem, rcut, ecut, primitive_type, 
                                        initializer, orbparam, mode)
        self.assertIsInstance(cascade, OrbgenCascade)
        
        orbparam = [{'nzeta': [2, 2, 2], 
                     'folders': [os.path.join(self.testfiles, f'Si-dimer-{bl}-7au')
                                 for bl in ['1.75', '2.00']], 
                     'nbnds': [8, 8], 
                     'iorb_frozen': None}]
        cascade_new = DeriveCascadeInstance(elem, rcut, ecut, primitive_type, 
                                            cascade, orbparam)
        self.assertIsInstance(cascade_new, OrbgenCascade)
        self.assertEqual(len(cascade_new.orbitals_), 2)
        self.assertEqual(cascade_new.optimized_, [False, False])

if __name__ == '__main__':
    unittest.main(exit=True)