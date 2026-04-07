'''
Concepts
--------
Interface of ABACUS module for ABACUS-ORBGEN
'''

# in-built modules
import os
import unittest
import uuid
import logging

# local modules
from SIAB.driver.control import OrbgenAssert
from SIAB.io.convention import dft_folder
from SIAB.abacus.io import dftparam_to_text, structure_to_text, ABACUS_PARAMS
from SIAB.abacus.utils import DuplicateCheck, autoset
from SIAB.abacus.blscan import blgen
from SIAB.spillage.radial import _nbes
from SIAB.spillage.datparse import read_input_script

def _build_case(proto, 
                pertkind, 
                pertmag, 
                atomspecies,
                dftshared, 
                dftspecific,
                **kwargs):
    '''build the single dft calculation case

    Parameters
    ----------
    proto : str
        the prototype of the structure
    pertkind : str
        the kind of perturbation
    pertmag : float
        the magnitude of perturbation
    atomspecies : dict
        the atom species in the structure, whose keys are element symbols,
        values are 'pp' and 'orb', indicating the pseudopotential and orbital
        file paths
    dftshared : dict
        the shared dft calculation parameters, will be overwritten by the
        dftspecific
    dftspecific : dict
        the specific dft calculation parameters

    Returns
    -------
    str|Nonetype
        if this job folder is not a duplicated one, return the folder, otherwise
        return None
    '''
    from os.path import basename, dirname, abspath
    
    elem = list(atomspecies.keys())
    OrbgenAssert(len(elem) == 1, 
                 f'only one element is supported: {elem}',
                 NotImplementedError)
    OrbgenAssert(pertkind == 'stretch',
                 f'only stretch perturbation is supported: {pertkind}',
                 NotImplementedError)
    
    elem = elem[0]
    rcut = None if dftshared.get('basis_type', 'jy') == 'pw' \
        else dftshared.get('bessel_nao_rcut')
    
    # INPUT
    dftparam = dftshared|dftspecific # overwrite the shared parameters
    dftparam = autoset(dftparam)
    dftparam |= {
        'pseudo_dir' : abspath(dirname(atomspecies[elem].get('pp'))),
        'orbital_dir': abspath(dirname(atomspecies[elem].get('orb', '.')))
    }

    folder = dft_folder(elem, proto, pertmag, rcut)

    if DuplicateCheck(folder, dftparam):
        logging.info(f'duplicated dft calculation: {folder}, skip.')
        return None
    
    logging.info(f'generate/overwrite dft calculation in folder: {folder}')
    os.makedirs(folder, exist_ok = True)

    with open(os.path.join(folder, 'INPUT'), 'w') as f:
        f.write(dftparam_to_text(dftparam))
        
    # STRU
    pp = basename(atomspecies[elem].get('pp'))
    orb = None if atomspecies[elem].get('orb') is None \
        else basename(atomspecies[elem].get('orb'))
    # the structure -> STRU file conversion can be more general
    # once there is the Cell instance and a method to convert it to STRU
    pertgeom_ = structure_to_text(
        shape=proto, 
        element=elem, 
        mass=1, 
        fpseudo=pp, 
        lattice_constant=kwargs.get('celldm', 30),
        bond_length=pertmag,
        nspin=dftparam.get('nspin', 1),
        forb=orb)[0] # discard the other return `natoms`
    with open(os.path.join(folder, 'STRU'), 'w') as f:
        f.write(pertgeom_)
    
    # INPUTw
    if dftparam.get('basis_type', 'jy') == 'pw':
        with open(os.path.join(folder, 'INPUTw'), 'w') as f:
            f.write('WANNIER_PARAMETERS\nout_spillage 2\n')
    return folder

def _filter_coef(coef, nzeta):
    '''always for development use, only get the orbitals that satisfy the
    filter condition.
    
    Parameters
    ----------
    coef : list
        the coefficients of the orbitals of one atomtype: [l][z][q]
    nzeta : list
        the number of zeta for each l
    
    Returns
    -------
    list
        the filtered coefficients
    '''
    return [[] if nz == 0 else ([coefl[-1]] if nz == 1 else coefl) 
            for coefl, nz in zip(coef, nzeta)]

def _cal_nzeta(rcut, ecut, lmaxmax, less_dof):
    '''calculate the number of zeta for each l
    
    Parameters
    ----------
    rcut : float
        the cutoff radius in a.u.
    ecut : float|str
        the kinetic energy cutoff in Ry, or a string startswith '='/'<='
    lmaxmax : int
        the maximum angular momentum, or a string startswith '='/'<='
    less_dof : int
        the degree of freedom that to be reduced
    
    Returns
    -------
    list
        the number of zeta for each l
    '''

    nq = 1 if isinstance(ecut, str) and ecut.startswith('=') else -1
    nl = 1 if isinstance(lmaxmax, str) and lmaxmax.startswith('=') else -1
    lmaxmax = int(lmaxmax.split('=')[-1]) if isinstance(lmaxmax, str) else lmaxmax
    ecut = float(ecut.split('=')[-1]) if isinstance(ecut, str) else ecut
    
    def nbes(l, rcut, ecut):
        if nl == 1:
            return 0 if l != lmaxmax else (_nbes(l, rcut, ecut) - less_dof if nq != 1 else 1)
        if nq == 1:
            return 1
        return _nbes(l, rcut, ecut) - less_dof
    
    return [nbes(l, rcut, ecut) for l in range(lmaxmax + 1)], ecut, lmaxmax

def _build_atomspecies(elem,
                       pp,
                       ecut = None,
                       rcut = None,
                       lmaxmax = None,
                       primitive_type = 'reduced',
                       orbital_dir = None,
                       norder_smooth = 1):
    '''build the atomspecies dictionary. Once ecut, rcut and lmaxmax
    all are provided, will generate the orbital file path
    
    Parameters
    ----------
    elem : str
        the element symbol
    pp : str
        the pseudopotential file path
    ecut : float|str
        the kinetic energy cutoff in Ry for spherical Bessel function, 
        or a string startswith '=', which means the exact value
    rcut : float
        the cutoff radius in a.u. for primtive jy
    lmaxmax : int|str
        the maximum angular momentum for generating the orbital file, 
        or a string startswith '=', which means the exact value
    primitive_type : str
        the type of primitive basis set
    orbital_dir : str
        where the generated orbital file is stored
    norder_smooth : int
        the highest order of derivative that is zero at the boundary. For more information,
        please refer to work https://journals.aps.org/prb/abstract/10.1103/PhysRevB.88.085117
    
    Returns
    -------
    dict
        the atomspecies dictionary
    '''
    from SIAB.orb.orb import Orbital
    from SIAB.io.convention import orb as name_orb
    out = {'pp': os.path.abspath(pp)}
    orbital_dir = os.path.join(os.getcwd(), 'primitive_jy') \
        if orbital_dir is None else orbital_dir
    if all([x is not None for x in [ecut, rcut, lmaxmax]]):
        less_dof = 0 if primitive_type == 'normalized' else norder_smooth # DOF: degree of freedom
        nzeta, ecut_, lmax_ = _cal_nzeta(rcut, ecut, lmaxmax, less_dof)
        forb = os.path.join(orbital_dir, name_orb(elem, rcut, ecut_, nzeta))
        out['orb'] = os.path.abspath(forb)
        if os.path.exists(forb):
            return {elem: out}
        # otherwise...
        orb = Orbital(rcut, ecut, elem, [0]*(lmax_+1), primitive_type, None, None)
        orb.coef_ = _filter_coef(orb.coefgen('ones'), nzeta)
        _ = orb.to_griddata(dr=0.01, fn=forb)
    return {elem: out}

def _build_pw(atomspecies, geom, rcuts, param_general, param_specific, **kwargs):
    '''build abacus_pw calculation on one structural prototype, perturbaed by
    one pertkind with magnitude specified by pertmag. Generate overlap matrix
    <psi|jy> for jy primitive basis functions with cutoff radius rcuts.
    
    Parameters
    ----------
    atomspecies : list[dict]
        a list of atom species, each element contains at least its symbol
    geom : dict
        the geometry of the structure, should contain the following keys:
        proto, pertkind, pertmag.
    rcuts : list
        real space cutoff for orbital to generate. For pw calculation, will be
        used to generate the OVERLAP_S, OVERLAP_Q and OVERLAP_Sq tables.
    param_general : dict
        general setting of an abacus run
    param_specific : dict
        structure-specific dft parameters setting, will overwrite the `param_general`
        if there is keyword definition overlap
    
    Returns
    -------
    list[str]
        return all executed folders. If the job in folder has been completed
        in previous run, it will not be returned by this function.
    '''
    return [_build_case(proto=geom['proto'], 
                        pertkind=geom['pertkind'], 
                        pertmag=geom['pertmag'], 
                        atomspecies=_build_atomspecies(elem=atomspecies[0]['elem'], 
                                                       pp=param_general['pseudo_dir']), 
                        dftshared=param_general|{'bessel_nao_rcut': rcuts},
                        dftspecific=param_specific,
                        celldm=kwargs.get('celldm', 30))]

def _build_jy(atomspecies, geom, rcuts, param_general, param_specific, **kwargs):
    '''build abacus lcao (jy) calculation on one structural prototype, perturbaed by
    one pertkind with magnitude specified by pertmag. 
    
    Parameters
    ----------
    atomspecies : list[dict]
        a list of atom species, each element contains at least its symbol
    geom : dict
        the geometry of the structure, should contain the following keys:
        proto, pertkind, pertmag.
    rcuts : list
        real space cutoff for orbital to generate. For lcao calculation, will
        be used to generate primitive jy orbitals with these rcuts
    param_general : dict
        general setting of an abacus run
    param_specific : dict
        structure-specific dft parameters setting, will overwrite the `param_general`
        if there is keyword definition overlap
    
    Returns
    -------
    list[str]
        return all executed folders. If the job in folder has been completed
        in previous run, it will not be returned by this function.
    '''
    return [_build_case(proto=geom['proto'], 
                        pertkind=geom['pertkind'],
                        pertmag=geom['pertmag'],
                        atomspecies=_build_atomspecies(elem=atomspecies[0]['elem'], 
                                                       pp=param_general['pseudo_dir'], 
                                                       ecut=atomspecies[0]['ecutjy'], 
                                                       rcut=rcut, 
                                                       lmaxmax=param_specific.get('lmaxmax', 2)), 
                        dftshared=param_general|{'bessel_nao_rcut': rcut},
                        dftspecific=param_specific,
                        celldm=kwargs.get('celldm', 30)) 
            for rcut in rcuts]

def _build_pert(elem, proto, pertkind, pertmags, n = 5, dr_l = 0.2, dr_r = 0.5):
    '''generate the perturbation on proto if not set
    
    Parameters
    ----------
    elem : str
        the element symbol
    proto : str
        the prototype of the structure
    pertkind : str
        the kind of perturbation. Only 'stretch' is supported
    pertmags : list[float]|str
        the magnitude of perturbation. If it is a string, will be treated as
        'default' or 'scan'
    n : int
        the number of perturbation to generate
    dr_l : float
        the stepsize of the left side of the perturbation
    dr_r : float
        the stepsize of the right side of the perturbation
        
    Returns
    -------
    list[float]
        the perturbation magnitude
    '''
    OrbgenAssert(pertkind == 'stretch',
                 f'only stretch perturbation is supported: {pertkind}',
                 NotImplementedError)

    if isinstance(pertmags, str):
        logging.info(f'pertmags is `{pertmags}`, a series of pertmags will be generated')
        return blgen(elem, proto, n // 2, dr_l, dr_r)
    
    OrbgenAssert(isinstance(pertmags, list) and\
                 all([isinstance(pert, (int, float)) for pert in pertmags]),
                 f'pertmags should be a list of float: {pertmags}')
   
    return pertmags

def build_abacus_jobs(atomspecies,
                      rcuts,
                      dftparams,
                      geoms,
                      spill_guess: str = 'atomic',
                      **kwargs):
    '''build series of abacus jobs
    
    Parameters
    ----------
    atomspecies : list[dict]
        the definition of atom species required by the abacus calculation
    rcuts : list
        the cutoff radius for the jy orbitals
    dftparams : dict
        the dft calculation parameters (shared)
    geoms : list of dict
        the geometries of the structures, also contains specific dftparam
        to overwrite the shared dftparam
    spill_guess : str
        the guess for the spillage optimization

    Returns
    -------
    list[str]
        return all executed folders. If the job in folder has been completed
        in previous run, it will not be returned by this function.
    '''
    mybuilder = {'lcao': _build_jy, 'pw': _build_pw}[dftparams.get('basis_type', 'lcao')]
    myatoms = [{'elem': atomspecies[0]['elem'],
                'ecutjy': atomspecies[0]['ecutjy'],
                'rcutjy': rcuts,
                'fpsp': None,  # when will the atomspecies know its pp?
                'forb': None}] # a list of AtomSpecies instances would be better
    
    jobs = []
    for geom in geoms: # transverse the 'geoms' key in input file...
        # collect those parameters specified in the geom section, will overwrite those
        # in the dftparams
        dftspecific = {k: v for k, v in geom.items() if k in ABACUS_PARAMS}
        for pertmag in _build_pert(elem=myatoms[0]['elem'], 
            proto=geom['proto'], pertkind=geom['pertkind'], pertmags=geom['pertmags']): 
            # except exact values, should also support 'default' and 'scan'
            jobs += mybuilder(atomspecies=myatoms,
                              # a Cell instance would be better?
                              geom={'proto': geom['proto'], 
                                    'pertkind': geom['pertkind'], 
                                    'pertmag': pertmag}, 
                              rcuts=rcuts, 
                              param_general=dftparams, 
                              param_specific=dftspecific,
                              celldm=geom.get('celldm', 30))

    # then the initial guess
    logging.info('')
    if spill_guess == 'atomic':
        lmaxmax = int(max([geom.get('lmaxmax', 2) for geom in geoms]))
        jobs += mybuilder(atomspecies=myatoms,
                          geom={'proto': 'monomer', 
                                'pertkind': 'stretch', 
                                'pertmag': 0},
                          rcuts=rcuts, 
                          param_general=dftparams, 
                          param_specific={
                              'nbands': kwargs.get('__iop_spill_guess_atomic_nbands__', 69), 
                              'lmaxmax': lmaxmax
                          })
        logging.warning('AUTOSET: `spill_guess` is `atomic`, the number of bands is set '
                        'to 69. This may be larger than number of basis available.')
        logging.warning('If ABACUS fails with warning that "the number of bands is larger'
                        '...", please set the Internal OPtion (IOP) `__iop_spill_guess_at'
                        'omic_nbands__` to a smaller value. However, this may also result'
                        ' in insufficient number of bands to refer in orbital '
                        'initialization. In this case, try any one of the following: 1. '
                        'increase the number of basis functions, 2. set `spill_guess` to '
                        '`random`, 3. reduce the number of zeta functions for each l in '
                        '`orbitals` section.')
        logging.warning('69 is the half of the number of electrons that can fill 5g '
                        'orbitals. If you are sure that you need more bands, please set '
                        'the IOP to a larger value.')
    else:
        logging.info(f'`spill_guess` is not `atomic`: {spill_guess}.')
    return [job for job in jobs if job is not None]

def job_done(folder):
    '''check if the abacus calculation is completed
    
    Parameters
    ----------
    folder : str
        the folder of the abacus calculation
    
    Returns
    -------
    bool
        True if the calculation is completed
    '''
    dftparam = read_input_script(os.path.join(folder, 'INPUT'))
    suffix = dftparam.get('suffix', 'ABACUS')
    runtype = dftparam.get('calculation', 'scf')
    outdir = os.path.join(folder, f'OUT.{suffix}')
    runninglog = os.path.join(outdir, f'running_{runtype}.log')
    if os.path.exists(runninglog):
        with open(runninglog, 'r') as f:
            lines = f.readlines()
        for i in range(len(lines) - 1, -1, -1):
            if 'Finish Time' in lines[i]:
                return True
    return False

class TestAbacusApi(unittest.TestCase):
    def test_build_atomspecies(self):
        here = os.getcwd()
        atomspecies = _build_atomspecies('H', 'H.pp')
        self.assertEqual(atomspecies, {'H': {'pp': os.path.join(here, 'H.pp')}})

        orbital_dir = os.path.join(here, str(uuid.uuid4()))
        atomspecies = _build_atomspecies('H', 'H.upf', 100, 7, 1, 'reduced', orbital_dir)
        self.assertTrue('orb' in atomspecies['H'])
        self.assertTrue(os.path.exists(atomspecies['H']['orb']))
        os.remove(atomspecies['H']['orb'])
        os.system('rm -rf ' + orbital_dir)

    def test_build_case(self):
        dftparam = {}
        atomspecies = {'H': {'pp': 'pseudo/H.pp'}}
        folder = _build_case('monomer', 'stretch', 0, atomspecies, dftparam, {})
        self.assertTrue(os.path.exists(folder))
        self.assertEqual(os.path.basename(folder), 'H-monomer')
        self.assertTrue(os.path.exists(os.path.join(folder, 'INPUT')))
        self.assertTrue(os.path.exists(os.path.join(folder, 'STRU')))
        os.system('rm -rf ' + folder)

        dftparam = {'basis_type': 'lcao', 'bessel_nao_rcut': 6}
        atomspecies = {'H': {'pp': 'pseudo/H.pp'}}
        folder = _build_case('monomer', 'stretch', 0, atomspecies, dftparam, {})
        self.assertTrue(os.path.exists(folder))
        self.assertTrue(os.path.exists(os.path.join(folder, 'INPUT')))
        self.assertTrue(os.path.exists(os.path.join(folder, 'STRU')))
        self.assertEqual(os.path.basename(folder), 'H-monomer-6au')
        os.system('rm -rf ' + folder)

        dftparam = {'basis_type': 'pw', 'bessel_nao_rcut': 6}
        atomspecies = {'H': {'pp': 'pseudo/H.pp', 'orb': 'orb/H_gga_6au_100Ry_2s1p.orb'}}
        folder = _build_case('dimer', 'stretch', 1.1, atomspecies, dftparam, {'lmaxmax': 2})
        self.assertTrue(os.path.exists(folder))
        self.assertTrue(os.path.exists(os.path.join(folder, 'INPUT')))
        self.assertTrue(os.path.exists(os.path.join(folder, 'STRU')))
        self.assertEqual(os.path.basename(folder), 'H-dimer-1.10')
        os.system('rm -rf ' + folder)

        dftparam = {'basis_type': 'pw', 'bessel_nao_rcut': [6, 7]}
        atomspecies = {'H': {'pp': 'pseudo/H.pp', 'orb': 'orb/H_gga_6au_100Ry_2s1p.orb'}}
        folder = _build_case('dimer', 'stretch', 1.1, atomspecies, dftparam, {'lmaxmax': 2})
        self.assertTrue(os.path.exists(folder))
        self.assertTrue(os.path.exists(os.path.join(folder, 'INPUT')))
        self.assertTrue(os.path.exists(os.path.join(folder, 'STRU')))
        self.assertEqual(os.path.basename(folder), 'H-dimer-1.10')
        os.system('rm -rf ' + folder)

        dftparam = {'basis_type': 'lcao', 'bessel_nao_rcut': 6}
        atomspecies = {'H': {'pp': 'pseudo/H.pp', 'orb': 'orb/H_gga_6au_100Ry_2s1p.orb'}}
        folder = _build_case('dimer', 'stretch', 1.1, atomspecies, dftparam, {'lmaxmax': 2})
        self.assertTrue(os.path.exists(folder))
        self.assertTrue(os.path.exists(os.path.join(folder, 'INPUT')))
        self.assertTrue(os.path.exists(os.path.join(folder, 'STRU')))
        self.assertEqual(os.path.basename(folder), 'H-dimer-1.10-6au')
        os.system('rm -rf ' + folder)

    def test_build_abacus_jobs(self):
        dftparams = {'basis_type': 'pw', 'pseudo_dir': 'pseudo', 'ecutwfc': 100}
        geoms = [{'proto': 'dimer', 'pertkind': 'stretch', 'pertmags': [0.25], 'nbands': 10},
                 {'proto': 'trimer', 'pertkind': 'stretch', 'pertmags': [0.5, 1.0], 'nbands': 20}]
        jobs = build_abacus_jobs([{'elem': 'He', 'ecutjy': 60}], [6, 7], dftparams, geoms)
        self.assertEqual(len(jobs), 4) # including the initial guess
        for job in jobs:
            self.assertTrue(os.path.exists(job))
            os.system('rm -rf ' + job)
        self.assertSetEqual(set(jobs), {'He-dimer-0.25', 'He-trimer-0.50', 'He-trimer-1.00', 'He-monomer'})
        
        # test without the initial guess
        jobs = build_abacus_jobs([{'elem': 'He', 'ecutjy': 60}], [6, 7], dftparams, geoms, 'random')
        self.assertEqual(len(jobs), 3)
        for job in jobs:
            self.assertTrue(os.path.exists(job))
            os.system('rm -rf ' + job)
        self.assertSetEqual(set(jobs), {'He-dimer-0.25', 'He-trimer-0.50', 'He-trimer-1.00'})

        dftparams = {'basis_type': 'lcao', 'pseudo_dir': 'pseudo', 'ecutwfc': 100, 'bessel_nao_rcut': 8}
        jobs = build_abacus_jobs([{'elem': 'He', 'ecutjy': 60}], [6, 7], dftparams, geoms)
        self.assertEqual(len(jobs), 8)
        for job in jobs:
            self.assertTrue(os.path.exists(job))
            os.system('rm -rf ' + job)
        self.assertSetEqual(set(jobs), {'He-dimer-0.25-6au', 'He-dimer-0.25-7au', 
                                        'He-trimer-0.50-6au', 'He-trimer-0.50-7au', 
                                        'He-trimer-1.00-6au', 'He-trimer-1.00-7au', 
                                        'He-monomer-6au', 'He-monomer-7au'})
        # test without the initial guess
        jobs = build_abacus_jobs([{'elem': 'He', 'ecutjy': 60}], [6, 7], dftparams, geoms, 'random')
        self.assertEqual(len(jobs), 6)
        for job in jobs:
            self.assertTrue(os.path.exists(job))
            os.system('rm -rf ' + job)
        self.assertSetEqual(set(jobs), {'He-dimer-0.25-6au', 'He-dimer-0.25-7au', 
                                        'He-trimer-0.50-6au', 'He-trimer-0.50-7au', 
                                        'He-trimer-1.00-6au', 'He-trimer-1.00-7au'})

    def test_cal_nzeta(self):
        self.assertEqual(_cal_nzeta(6, 100, 2, 1)[0], [18, 17, 17])
        self.assertEqual(_cal_nzeta(6, '<=100', '<=2', 1)[0], [18, 17, 17])
        self.assertEqual(_cal_nzeta(6, '=100', 2, 1)[0], [1, 1, 1])
        self.assertEqual(_cal_nzeta(6, 100, '=2', 1)[0], [0, 0, 17])

if __name__ == '__main__':
    unittest.main()