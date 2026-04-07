'''this module is to read the input of ABACUS ORBGEN-v3.0 input script'''
# in-built modules
import json
import os
import logging
import unittest
import time

# local modules
from SIAB.abacus.io import ABACUS_PARAMS
from SIAB.driver.control import OrbgenAssertIn, OrbgenAssert

PROGRAM_HEADER = f'''
===============================================================================
    
    
       ******   ******** **       **       ****     **     **       *******  
      **////** **////// /**      /**      /**/**   /**    ****     **/////** 
     **    // /**       /**   *  /**      /**//**  /**   **//**   **     //**
    /**       /*********/**  *** /** *****/** //** /**  **  //** /**      /**
    /**       ////////**/** **/**/**///// /**  //**/** **********/**      /**
    //**    **       /**/**** //****      /**   //****/**//////**//**     ** 
     //******  ******** /**/   ///**      /**    //***/**     /** //*******  
      //////  ////////  //       //       //      /// //      //   ///////   
    
    Contracted Spherical Wave Numerical Atomic Orbital (CSW-NAO) generator
    Author: @kirk0830 @jinzx10
    Visit: https://github.com/kirk0830/ABACUS-CSW-NAO
    
    Based on ABACUS (Atomic-orbital Based Ab-initio Computation at UStc)
    in DeepModeling Open source Community
    Visit: https://github.com/deepmodeling/abacus-develop
    
>>> Program Start Time: {time.asctime()}
...
'''

PROGRAM_TAIL = f'''
...
>>> Program End Time: {time.asctime()}
===============================================================================
'''

def ParamAssert(params):
    '''check the correctness of the input parameters
    
    Parameters
    ----------
    params : dict
        the input parameters
    '''
    logging.info('checking the input parameters')
    COMPULSORY_ = ['abacus_command', 'pseudo_dir', 
        'element', 'bessel_nao_rcut', 'geoms', 'orbitals']
    logging.info(f'Defined compulsory keys are: {COMPULSORY_}')
    OrbgenAssertIn(COMPULSORY_, params, 
        lambda x: f'key {x} is missing in the input')
    # check if there is really pseudopotential point to the right directory
    OrbgenAssert(os.path.exists(params['pseudo_dir']), 
        f'pseudo_dir {params["pseudo_dir"]} does not exist',
        FileNotFoundError)
    logging.info('pseudopotential existence check passed')

    # check if bessel_nao_rcut is list of int
    OrbgenAssert(isinstance(params['bessel_nao_rcut'], list), 
        f'bessel_nao_rcut should be a list of int: {params["bessel_nao_rcut"]}')
    OrbgenAssert(all(isinstance(i, int) for i in params['bessel_nao_rcut']), 
        f'bessel_nao_rcut should be a list of int: {params["bessel_nao_rcut"]}')
    logging.info('rcut check passed')

    # check if geom is a list of dict
    OrbgenAssert(isinstance(params['geoms'], list), 
        f'geoms should be a list of dict: {params["geoms"]}')
    OrbgenAssert(all(isinstance(i, dict) for i in params['geoms']),
        f'geoms should be a list of dict: {params["geoms"]}')
    logging.info('geom type check passed')

    # check if every geom has at least keys: 'proto', 'pertkind', 'pertmags', 'lmaxmax'
    GEOM_COMPULSORY_ = ['proto', 'pertkind', 'pertmags', 'lmaxmax']
    for i, geom in enumerate(params['geoms']):
        OrbgenAssert(all(key in geom for key in GEOM_COMPULSORY_), 
            f'geom {i} does not have all the compulsory keys')
        GeomAssert(geom)
    logging.info('geom key&val check passed')

    # check if orbitals is a list of dict
    OrbgenAssert(isinstance(params['orbitals'], list),
        '`orbitals` section should be a list of dict')
    logging.info('orbitals type check passed')

    # check if each orbital has keys: 'nzeta', 'geoms', 'nbands' and 'checkpoint'
    ORBITAL_COMPULSORY_ = ['nzeta', 'geoms', 'nbands', 'checkpoint']
    for i, orb in enumerate(params['orbitals']):
        OrbgenAssert(all(key in orb for key in ORBITAL_COMPULSORY_),
            f'orbital {i} does not have all the compulsory keys')
        OrbitalAssert(orb)
    logging.info('orbital key&val check passed')

    ComprehensiveParamAssert(params)
    logging.info('comprehensive check passed')

    logging.info('ParamAssert passed')

def OrbitalAssert(orb):
    '''check the integrity of the orbital parameters
    
    Parameters
    ----------
    orb : dict
        the orbital parameters
    '''
    # check if nzeta is 'auto' or a list of int
    cond1 = isinstance(orb['nzeta'], list) and \
        all(isinstance(i, int) for i in orb['nzeta'])
    cond2 = isinstance(orb['nzeta'], str)
    OrbgenAssert(cond1 or cond2, 
        f'nzeta should be a list of int or a string: {orb["nzeta"]}')
    
    # check if geoms is a list of int (index of geoms defined in geoms section)
    cond = isinstance(orb['geoms'], list) and \
        all(isinstance(i, int) for i in orb['geoms'])
    OrbgenAssert(cond, f'geoms should be a list of int (the index \
of geoms defined in geoms section): {orb["geoms"]}')

    # check if nbands is a list of int or str
    cond1 = isinstance(orb['nbands'], list) and \
        all(isinstance(i, (int, str)) for i in orb['nbands'])
    cond2 = isinstance(orb['nbands'], (int, str))
    OrbgenAssert(cond1 or cond2,
        f'nbands should be one of the following: a list of int/str or \
a int/str: {orb["nbands"]}')

    # check if checkpoint is a int or None
    OrbgenAssert(isinstance(orb['checkpoint'], (int, type(None))),
        f'checkpoint should be a int or None: {orb["checkpoint"]}')


def GeomAssert(geom):
    '''check the integrity of the geom parameters
    
    Parameters
    ----------
    geom : dict
        the geom parameters
    '''
    import os
    import re
    # check if proto is a string
    OrbgenAssert(isinstance(geom['proto'], str), 'proto should be a string')
    # check if the proto is a file or a pre-defined string in the following
    # supported range: dimer, trimer, square, tetrahedron, octahedron, cube
    OrbgenAssert(geom['proto'] in ['dimer', 'trimer', 'square', 'tetrahedron',
        'octahedron', 'cube'] or os.path.exists(geom['proto']),
        f'proto should be a file or one of the following: dimer, trimer, \
square, tetrahedron, octahedron, cube: {geom["proto"]}')
    
    # check pertkind
    OrbgenAssert(geom.get('pertkind', 'stretch') in 
        ['stretch', 'shear', 'twist'],
        f'pertkind should be one of the following: stretch, shear, twist: \
{geom["pertkind"]}')
    # temporary
    OrbgenAssert(geom.get('pertkind', 'stretch') == 'stretch',
        '`pertkind != stretch` functionality is not supported yet', 
        NotImplementedError)

    # check if pertmags is list[int|float], or a string 'auto'
    OrbgenAssert('pertmags' in geom, 'pertmags should be defined')
    cond1 = isinstance(geom['pertmags'], list) and \
        all(isinstance(i, (int, float)) for i in geom['pertmags'])
    cond2 = geom['pertmags'] == 'auto'
    OrbgenAssert(cond1 or cond2, 
        f'pertmags should be a list of int/float or `auto`: {geom["pertmags"]}')

    # check if lmaxmax is a non-negative int
    cond1 = isinstance(geom['lmaxmax'], int) and geom['lmaxmax'] >= 0
    cond2 = isinstance(geom['lmaxmax'], str) and \
        re.match(r'\=\d+', geom['lmaxmax'])
    OrbgenAssert(cond1 or cond2, 
        f'lmaxmax should be a non-negative int or a string like \
`=3` (development use): {geom["lmaxmax"]}')

    # check if nbands is a positive int
    OrbgenAssert(isinstance(geom['nbands'], int) and geom['nbands'] > 0, 
        f'nbands should be a positive int: {geom["nbands"]}')

    # check if celldm is a positive int or float
    OrbgenAssert(isinstance(geom.get('celldm', 1.0), (int, float)) and \
        geom.get('celldm', 1.0) > 0, 
        f'celldm should be a positive int or float: {geom.get("celldm", 1.0)}')

def ComprehensiveParamAssert(params):
    '''check the correctness of the input parameters across sections
    
    Parameters
    ----------
    params : dict
        the input parameters
    '''
    # check if more bands are required in orbital section than in geom section
    for i, orb in enumerate(params['orbitals']):
        spill_nbnds = orb['nbands'] # check `nbands` for each orb
        spill_nbnds = [spill_nbnds] * len(orb['geoms']) \
            if not isinstance(spill_nbnds, list) else spill_nbnds
        for temp, j in zip(spill_nbnds, orb['geoms']): # check all geoms the orb requires
            # j is the index of geoms in the geom section...
            if isinstance(temp, str):
                logging.info(f'orbital {i} requires `{temp}` bands for geom {j}')
                continue
            OrbgenAssert(j < len(params['geoms']), 
                f'orbital {i} requires geom {j} which is not defined')
            OrbgenAssert(params['geoms'][j]['nbands'] >= temp,
                f'orbital {i} requires more bands than geom {j} has')
    logging.info('nbands check passed')

def group(params):
    '''parameters defined in input script are for different fields,
    this function is to group them into different sets.
    
    Parameters
    ----------
    params : dict
        the input parameters

    Returns
    -------
    dict, dict, dict, dict
        the global parameters, the DFT parameters, the spillage parameters, 
        the compute parameters
    '''
    GLOBAL = ['element', 'bessel_nao_rcut']
    DFT = [k for k in ABACUS_PARAMS if k != 'bessel_nao_rcut']
    COMPUTE = ['environment', 'mpi_command', 'abacus_command']
    SPILLAGE = ['fit_basis', 'primitive_type', 'optimizer', 'verbose', 
                'max_steps', 'spill_guess', 'nthreads_rcut', 'geoms', 'orbitals',
                'ecutjy']
    
    # extract all relevant parameters and delete those unspecified
    dftparams = {key: params.get(key) for key in DFT}
    dftparams = {k: v for k, v in dftparams.items() if v is not None}
    
    optimizer = {k: v for k, v in params.items() 
                 if k.startswith('scipy.') or k.startswith('torch.')}
    spillparams = {key: params.get(key) for key in SPILLAGE if key in params}|optimizer
    spillparams['ecutjy'] = spillparams.get('ecutjy', dftparams.get('ecutwfc'))
    
    glbparams = {key: params.get(key) for key in GLOBAL}
    
    compute = {key: params.get(key) for key in COMPUTE}

    iop = {k: v for k, v in params.items() if k.startswith('__iop')}
    return glbparams, dftparams, spillparams, compute, iop

def read(fn):
    '''read the input of ABACUS ORBGEN-v3.0 input script
    
    Parameters
    ----------
    fn : str
        the filename of the input script
    
    Returns
    -------
    dict, dict, dict, dict
        the global parameters, the DFT parameters, the spillage parameters,
        the compute parameters. For detailed explanation, see group function
    '''
    
    with open(fn) as f:
        params = json.load(f)
    
    try:
        ParamAssert(params)
    except Exception as e:
        logging.error(f'error in checking the input: {e}')
        raise e
    
    return group(params)

def orb_link_geom(indexes, geoms):
    '''link the indexes of geoms to proper geom parameters
    
    Parameters
    ----------
    indexes : list of int
        the indexes of geoms
    geoms : list of dict
        the geom parameters
    
    Returns
    -------
    list of dict
        the geom parameters
    '''
    return [{k: v for k, v in geoms[i].items() 
             if k in ['proto', 'pertkind', 'pertmags']} 
            for i in indexes]

class TestReadv3p0(unittest.TestCase):

    def test_OrbitalAssert(self):
        orb = {'nzeta': 'auto', 'geoms': [0, 1], 'nbands': [1, 2], 'checkpoint': 1}
        OrbitalAssert(orb)
        orb = {'nzeta': [1, 2], 'geoms': [0, 1], 'nbands': [1, 2], 'checkpoint': 1}
        OrbitalAssert(orb)
        orb = {'nzeta': [1, 2], 'geoms': [0, 1], 'nbands': [1, 2], 'checkpoint': None}
        OrbitalAssert(orb)
        orb = {'nzeta': 'auto', 'geoms': [0, 1], 'nbands': [1, 2], 'checkpoint': '1'}
        with self.assertRaises(ValueError):
            OrbitalAssert(orb)
    
    def test_GeomAssert(self):
        correct = {'proto': 'dimer', 
                   'pertkind': 'stretch', 
                   'pertmags': [1, 2], 
                   'nbands': 1,
                   'lmaxmax': 1}
        crazy = ['my little dog is', 3.14, {'age': 'years old'}, {'has'},
                 ['hobby'], -1]
        GeomAssert(correct) # al-right
        for crazy_proto in crazy:
            with self.assertRaises(ValueError):
                GeomAssert(correct|{'proto': crazy_proto})
        for crazy_pertkind in crazy:
            with self.assertRaises(ValueError):
                GeomAssert(correct|{'pertkind': crazy_pertkind})
        for crazy_pertmags in crazy:
            with self.assertRaises(ValueError):
                GeomAssert(correct|{'pertmags': crazy_pertmags})
        for crazy_lmaxmax in crazy:
            with self.assertRaises(ValueError):
                GeomAssert(correct|{'lmaxmax': crazy_lmaxmax})
        for crazy_nbands in crazy:
            with self.assertRaises(ValueError):
                GeomAssert(correct|{'nbands': crazy_nbands})
    
    def test_ParamAssert(self):
        here = os.path.dirname(os.path.dirname(__file__))
        pporb = os.path.join(os.path.dirname(here), 'tests', 'pporb')
        fpseudo = os.path.join(pporb, 'Si_ONCV_PBE-1.0.upf')

        params = {'abacus_command': 'test', 
                  'pseudo_dir': fpseudo, 
                  'element': 'test', 
                  'bessel_nao_rcut': [1, 2],
                  'geoms': [{'proto': 'test', 
                             'pertkind': 'test', 
                             'pertmags': [1, 2], 
                             'lmaxmax': 1}],
                  'orbitals': [{'nzeta': 'auto', 
                                'geoms': [0, 1], 
                                'nbands': [1, 2], 
                                'checkpoint': 1}]}
        with self.assertRaises(ValueError):
            ParamAssert(params)
        params = {'abacus_command': 'test', 
                  'pseudo_dir': 'test', 
                  'element': 'test', 
                  'bessel_nao_rcut': [1, 2],
                  'geoms': [{'proto': 'test', 
                             'pertkind': 'test', 
                             'pertmags': [1, 2], 
                             'lmaxmax': 1}],
                  'orbitals': [{'nzeta': 'auto', 
                                'geoms': [0, 1], 
                                'nbands': [1, 2], 
                                'checkpoint': 1}]}
        with self.assertRaises(FileNotFoundError):
            ParamAssert(params)
        params = {'abacus_command': 'test', 'pseudo_dir': fpseudo, 'element': 'test', 'bessel_nao_rcut': [1, 2],
                  'geoms': [{'proto': 'test', 'pertkind': 'test', 'pertmags': [1, 2], 'lmaxmax': 1}],
                  'orbitals': [{'nzeta': 'auto', 'geoms': [0, 1], 'nbands': [1, 2], 'checkpoint': 1.0}]}
        with self.assertRaises(ValueError):
            ParamAssert(params)
        params = {'abacus_command': 'test', 'pseudo_dir': fpseudo, 'element': 'test', 'bessel_nao_rcut': [1, 2],
                  'geoms': [{'proto': 'test', 'pertkind': 'test', 'pertmags': [1, 2], 'lmaxmax': 1}],
                  'orbitals': [{'nzeta': 'auto', 'geoms': [0, 1], 'nbands': [1, 2], 'checkpoint': '1'}]}
        with self.assertRaises(ValueError):
            ParamAssert(params)
        params = {'abacus_command': 'test', 'pseudo_dir': fpseudo, 'element': 'test', 'bessel_nao_rcut': [1, 2],
                  'geoms': [{'proto': 'test', 'pertkind': 'test', 'pertmags': [1, 2], 'lmaxmax': 1}],
                  'orbitals': [{'nzeta': 'auto', 'geoms': [0, 1], 'nbands': [1, 2], 'checkpoint': None}]}
        with self.assertRaises(ValueError):
            ParamAssert(params)
        params = {'abacus_command': 'test', 'pseudo_dir': fpseudo, 'element': 'test', 'bessel_nao_rcut': [1, 2],
                  'geoms': [{'proto': 'test', 'pertkind': 'test', 'pertmags': [1, 2], 'lmaxmax': 1}],
                  'orbitals': [{'nzeta': 'auto', 'geoms': [0, 1], 'nbands': [1, 2], 'checkpoint': '1'}]}
        with self.assertRaises(ValueError):
            ParamAssert(params)

if __name__ == '__main__':
    unittest.main()