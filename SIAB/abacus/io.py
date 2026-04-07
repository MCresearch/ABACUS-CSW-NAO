'''
Concepts
--------
this module defines ABACUS related (input and output) operations.
'''
# in-built modules
import re
import os
import logging
import unittest

# third-party modules
import numpy as np

# local modules
from SIAB.driver.control import OrbgenAssert, OrbgenAssertIn
from SIAB.supercomputing.op import op as envop
from SIAB.data.structures import monomer, dimer, trimer, tetrahedron,\
    square, triangular_bipyramid, octahedron, cube

BLSCAN_WARNMSG = """
WARNING: since SIAB version 2.1(2024.6.3), the original functionality invoked by value \"auto\" is replaced by 
        \"scan\", and for dimer the \"auto\" now will directly use in-built dimer database if available, otherwise will 
         fall back to \"scan\". This warning will be print everytime if \"auto\" is used. To disable this warning, specify 
         directly the \"bond_lengths\" in any one of following ways:
         1. a list of floats, e.g. [2.0, 2.5, 3.0]
         2. a string \"default\", which will use default bond length for dimer, and scan for other shapes, for other shapes, will
            fall back to \"scan\".
         3. a string \"scan\", which will scan bond lengths for present shape.
"""
##############################################
#         ABACUS output parser               #
##############################################
def read_energy(folder: str,
                suffix: str,
                calculation: str = "scf"):
    
    suffix = suffix or "ABACUS"
    frunninglog = os.path.join(folder, f"OUT.{suffix}", 
                               f"running_{calculation}.log")
    OrbgenAssert(os.path.exists(frunninglog),
                 f"running log {frunninglog} not found.", 
                 FileNotFoundError)

    with open(frunninglog, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    for l in lines:
        if l.startswith("!FINAL_ETOT_IS"):
            energy = float(l.split()[-2])
            return energy
    OrbgenAssert(False, f'energy not found in {frunninglog}')

def read_natom(folder: str,
               suffix: str,
               calculation: str = "scf"):
    suffix = suffix or "ABACUS"
    frunninglog = os.path.join(folder, f"OUT.{suffix}", 
                               f"running_{calculation}.log")
    OrbgenAssert(os.path.exists(frunninglog),
                 f"running log {frunninglog} not found.", 
                 FileNotFoundError)

    with open(frunninglog, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    for l in lines:
        if l.startswith("TOTAL ATOM NUMBER"):
            natom = int(l.split()[-1])
            return natom
    OrbgenAssert(False, f'natom not found in {frunninglog}')

##############################################
#         input files preparation            #
##############################################

def structure_to_text(shape: str, element: str, mass: float, fpseudo: str, 
         lattice_constant: float, bond_length: float, nspin: int,
         forb = None):
    """generate structure"""
    # add a warning message here, to avoid the unexpect overlap between PBC images...
    # this is only possible for an LCAO run
    if forb is not None:
        m = re.match(r'^([A-Z][a-z]?)_gga_(\d+(\.\d+)?)au_(\d+(\.\d+)?)Ry_(\d+[spdfghi])+.orb$', forb)
        assert m, f'The parameter `forb` is specified but not in standard format? {forb}'
        elem, rcut = m.groups()[:2]
        assert elem == element
        rcut = float(rcut)
        # rcut and lattice_constant are in Bohr, the bond_length is in Angstrom
        lattmin = 2*rcut + bond_length * 1.8897259886 # in Bohr
        OrbgenAssert(lattmin < lattice_constant,
            'The lattice constant is not large enough to avoid the orbital overlap ' \
            f'between PBC images, minimal suggested: {int(np.ceil(lattmin))}. '\
            'Set an appropriate value in `geom` section with parameter `celldm`.')
    # if everything looks fine, we continue
    if shape == "monomer":
        return monomer(element, mass, fpseudo, lattice_constant, nspin, forb), 1
    elif shape == "dimer":
        return dimer(element, mass, fpseudo, lattice_constant, bond_length, nspin, forb), 2
    elif shape == "trimer":
        return trimer(element, mass, fpseudo, lattice_constant, bond_length, nspin, forb), 3
    elif shape == "tetrahedron":
        return tetrahedron(element, mass, fpseudo, lattice_constant, bond_length, nspin, forb), 4
    elif shape == "square":
        return square(element, mass, fpseudo, lattice_constant, bond_length, nspin, forb), 4
    elif shape == "triangular_bipyramid":
        return triangular_bipyramid(element, mass, fpseudo, lattice_constant, bond_length, nspin, forb), 5
    elif shape == "octahedron":
        return octahedron(element, mass, fpseudo, lattice_constant, bond_length, nspin, forb), 6
    elif shape == "cube":
        return cube(element, mass, fpseudo, lattice_constant, bond_length, nspin, forb), 8
    else:
        OrbgenAssert(False, 
                     "shape %s is not supported"%shape,
                     NotImplementedError)

def KPOINTS():
    """For ABACUS-orbitals numerical orbitals generation workflow specifically"""
    return "K_POINTS\n0\nGamma\n1 1 1 0 0 0\n"

def autoset(dftparam: dict, **kwargs):
    '''
    auto-set some parameters of dft calculation, for consistency and better
    performance.
    
    Parameters
    ----------
    dftparam: dict
        the dft parameters
    kwargs: dict
        the parameters that will not use the default strategy built-in this
        function, instead, use the value in kwargs
        
    Returns
    -------
    out: dict
        the updated dft parameters
    '''
    OrbgenAssertIn(dftparam.keys(), ABACUS_PARAMS,
        lambda x: f'{x} is not found in abacus parameters.')
    OrbgenAssertIn(kwargs.keys(), ABACUS_PARAMS,
        lambda x: f'{x} is not found in abacus parameters.')
    OrbgenAssert(len(set(dftparam.keys()) & set(kwargs.keys())) == 0,
        'There are parameters defined twice in dftparam and kwargs.')
    base = {
        'suffix': 'ABACUS', 'stru_file': 'STRU', 'kpoint_file': 'KPT', 
        'wannier_card': 'INPUTw', 'pseudo_dir': './', 'calculation': 'scf',
        'basis_type': 'pw', 'ecutwfc': '100', 'ks_solver': 'dav', 
        'nbands': 'auto', 'scf_thr': '1.0e-7', 'scf_nmax': '9000',
        'smearing_method': 'gauss', 'smearing_sigma': '0.015',
        'mixing_type': 'broyden', 'mixing_beta': '0.8', 'mixing_ndim': '8', 
        'mixing_gg0': '0','ntype': '1', 'nspin': '1','lmaxmax': '4', 
        'bessel_nao_rcut': '10', 'gamma_only': '1', 'printe': '1', 
        'out_chg': '-1'
    }
    out = base.copy()
    __spin_polarised__ = {
        'nspin': '2', 'mixing_beta': '0.4', 'mixing_beta_mag': '0.4'
    }
    if dftparam.get('nspin', 1) == 2:
        out.update(__spin_polarised__)
    __lcao__ = {
        'ks_solver': 'genelpa', 'out_mat_hs': '1 12', 'out_mat_tk': '1 12', 
        'out_wfc_lcao': '1'
    }
    if dftparam.get('basis_type', 'pw') != 'pw':
        out.update(__lcao__)
    
    # imposing all user settings
    out.update(dftparam)
    out.update(kwargs)
    return out

def dftparam_to_text(dftparam, **kwargs):
    '''convert the dft parameters to the ABACUS INPUT file'''
    if kwargs:
        logging.warning('the second arg of function `dftparam_to_text` is deprecated')
    out = "INPUT_PARAMETERS\n"
    for key, value in dftparam.items():
        if value is None:
            continue
        if isinstance(value, list):
            value = ' '.join([str(v) for v in value])
        out += f'{key:<20} {str(value)}\n'
    return out

##############################################
#              file operations               #
##############################################
def configure(input_setting: dict,
              stru_setting: dict):
    """generate input files for orbital generation in present folder
    
    input_settings: dict, INPUT settings for ABACUS
    stru_settings: dict, structure settings for ABACUS

    Return:
        folder: str, a string used to distinguish different orbital generation tasks
    Details:
        in `stru_settings`, at least contain `shape`, `element`, `fpseudo` and `bond_length`
        information.
    """
    import os
    # CHECK
    OrbgenAssertIn(["element", "shape", "fpseudo", "bond_length"], stru_setting.keys(),
                   lambda x: f'{x} is not found in stru_setting')
    
    # mostly value will not be None, except the case the monomer is included to be referred
    # in initial guess of coefficients of sphbes
    keys_in_foldername = ["element", "shape"]
    keys_in_foldername.append("bond_length") if stru_setting["shape"] != "monomer" else None
    # because bond_length is not necessary for monomer

    def write(suffix, inp, stru):
        inp = dftparam_to_text(inp, suffix)
        stru = structure_to_text(**stru)
        with open("INPUT-"+suffix, "w") as f:
            f.write(inp)
        with open("STRU-"+suffix, "w") as f:
            f.write(stru[0])
        with open("KPT-"+suffix, "w") as f:
            f.write(KPOINTS())
        with open("INPUTw", "w") as f:
            f.write("WANNIER_PARAMETERS\n")
            f.write("out_spillage 2\n")
        return suffix

    folder = f"{stru_setting['element']}-{stru_setting['shape']}"
    folder += "-%3.2f"%stru_setting["bond_length"] if stru_setting["shape"] != "monomer" else ""
    orbital_dir = input_setting.get("orbital_dir")
    if orbital_dir is not None:
        forbs = [os.path.basename(f) for f in orbital_dir]
        dirs = [os.path.dirname(d) for d in orbital_dir]
        assert len(set(dirs)) == 1, "all temporary jybasis files is set to the same directory"
        assert len(forbs) == len(input_setting["bessel_nao_rcut"]), "number of forbs should be the same as bessel_nao_rcut"
        
        for forb, rcut in zip(forbs, input_setting["bessel_nao_rcut"]):
            inp = input_setting.copy()
            inp.update({"bessel_nao_rcut": rcut, "orbital_dir": dirs[0]})
            stru_setting["forb"] = forb
            folder_rcut = "-".join([folder, str(rcut) + "au"])
            yield write(folder_rcut, inp, stru_setting)
    else:
        yield write(folder, input_setting, stru_setting)

def archive(footer: str = "", env: str = "local"):
    """mkdir and move correspnding input files to folder"""
    headers = ["INPUT", "STRU", "KPT"]
    OrbgenAssert(footer != "", "footer is not specified")

    envop("mkdir", footer, additional_args=["-p"], env=env)
    for header in headers:
        if header == "INPUT":
            envop("mv", "%s-%s"%(header, footer), "%s/INPUT"%(footer), env=env)
        else:
            envop("mv", "%s-%s"%(header, footer), "%s/"%(footer), env=env)
    envop("mv", "INPUTw", "%s/INPUTw"%(footer), env=env)

def parse_abacus_dftparam(folder: str = "") -> dict:
    """parse ABACUS INPUT file, return a dict"""
    if folder.startswith("INPUT_PARAMETERS"):
        lines = folder.split("\n")
    else:
        with open(folder+"/INPUT", "r") as f:
            lines = f.readlines()

    pattern = r"^(\s*)([\w_]+)(\s+)([^\#]+)(.*)$"
    result = {}
    for line in lines:
        if line == "INPUT_PARAMETERS":
            continue
        else:
            match = re.match(pattern, line.strip())
            if match is not None:
                result[match.group(2)] = match.group(4)
    return result

DEFAULT_BOND_LENGTH = {
"dimer": {'H': [0.6, 0.75, 0.9, 1.2, 1.5], 'He': [1.25, 1.75, 2.4, 3.25], 
'Li': [1.5, 2.1, 2.5, 2.8, 3.2, 3.5, 4.2], 'Be': [1.75, 2.0, 2.375, 3.0, 4.0], 'B': [1.25, 1.625, 2.5, 3.5], 
'C': [1.0, 1.25, 1.5, 2.0, 3.0], 'N': [1.0, 1.1, 1.5, 2.0, 3.0], 'O': [1.0, 1.208, 1.5, 2.0, 3.0], 
'F': [1.2, 1.418, 1.75, 2.25, 3.25], 'Fm': [1.98, 2.375, 2.75, 3.25, 4.25], 'Md': [2.08, 2.5, 3.0, 3.43, 4.25], 
'No': [2.6, 3.125, 3.75, 4.27, 5.0], 'Ne': [1.5, 1.75, 2.25, 2.625, 3.0, 3.5], 'Na': [2.05, 2.4, 2.8, 3.1, 3.3, 3.8, 4.3], 
'Mg': [2.125, 2.375, 2.875, 3.375, 4.5], 'Al': [2.0, 2.5, 3.0, 3.75, 4.5], 'Si': [1.75, 2.0, 2.25, 2.75, 3.75], 
'P': [1.625, 1.875, 2.5, 3.25, 4.0], 'S': [1.6, 1.9, 2.5, 3.25, 4.0], 'Cl': [1.65, 2.0, 2.5, 3.25, 4.0], 
'Ar': [2.25, 2.625, 3.0, 3.375, 4.0], 'K': [1.8, 2.6, 3.4, 3.8, 4.0, 4.4, 4.8], 'Ca': [2.5, 3.0, 3.5, 4.0, 5.0], 
'Sc': [1.75, 2.15, 2.75, 3.5, 4.5], 'Ti': [1.6, 1.85, 2.5, 3.25, 4.25], 'V': [1.45, 1.65, 2.25, 3.0, 4.0], 
'Cr': [1.375, 1.55, 2.0, 2.75, 3.75], 'Mn': [1.4, 1.6, 2.1, 2.75, 3.75], 'Fe': [1.45, 1.725, 2.25, 3.0, 4.0], 
'Co': [1.8, 2.0, 2.5, 3.5], 'Ni': [1.65, 2.0, 2.5, 3.0, 4.0], 'Cu': [1.8, 2.2, 3.0, 4.0], 
'Zn': [2.0, 2.3, 2.85, 3.5, 4.25], 'Ga': [1.85, 2.1, 2.45, 3.0, 4.0], 'Ge': [1.8, 2.0, 2.35, 3.0, 4.0], 
'As': [1.75, 2.1, 2.5, 3.0, 4.0], 'Se': [1.85, 2.15, 2.5, 3.0, 4.0], 'Br': [1.9, 2.25, 2.75, 3.25, 4.0], 
'Kr': [2.4, 3.0, 3.675, 4.25, 5.0], 'Rb': [2.45, 3.0, 4.0, 5.0], 'Sr': [2.75, 3.5, 4.4, 5.0], 
'Y': [2.125, 2.5, 2.875, 3.25, 4.0, 5.0], 'Zr': [1.9, 2.25, 3.0, 4.0], 'Nb': [1.75, 2.05, 2.4, 3.0, 4.0], 
'Mo': [1.675, 1.9, 2.375, 3.0, 4.0], 'Tc': [1.7, 1.915, 2.375, 3.0, 4.0], 'Ru': [1.725, 1.925, 2.375, 3.0, 4.0], 
'Rh': [1.8, 2.1, 2.5, 3.0, 4.0], 'Pd': [2.0, 2.275, 2.75, 3.75], 'Ag': [2.1, 2.45, 3.0, 4.0], 
'Cd': [2.15, 2.5, 3.1, 4.0, 5.0], 'In': [2.15, 2.5, 3.0, 3.75, 4.75], 'Sn': [2.1, 2.4, 3.75, 3.5, 4.5], 
'Sb': [2.1, 2.5, 3.0, 3.5, 4.5], 'Te': [2.15, 2.55, 3.1, 3.6, 4.5], 'I': [2.22, 2.65, 3.25, 4.25], 
'Xe': [3.0, 3.5, 4.06, 4.5, 5.25], 'Cs': [2.7, 3.5, 4.5, 5.5], 'Ba': [2.65, 3.0, 3.5, 4.4, 5.5], 
'La': [2.2, 2.6, 3.25, 4.0, 5.0], 'Ce': [2.0, 2.375, 2.875, 3.5, 4.5], 'Pr': [1.9, 2.25, 2.75, 3.5, 4.5], 
'Nd': [1.8, 2.125, 2.625, 3.375, 4.5], 'Pm': [1.775, 2.05, 2.5, 3.25, 4.25], 'Sm': [1.775, 2.05, 2.5, 3.25, 4.25], 
'Eu': [1.775, 2.075, 2.5, 3.25, 4.25], 'Gd': [1.8, 2.11, 2.625, 3.375, 4.1, 5.0], 'Tb': [1.825, 2.16, 2.625, 3.375, 4.1, 5.0], 
'Dy': [1.85, 2.24, 2.625, 3.375, 4.1, 5.0], 'Ho': [1.93, 2.375, 3.0, 4.1, 5.0], 'Er': [2.025, 2.5, 3.125, 4.1, 5.0], 
'Tm': [2.2, 2.625, 3.25, 4.1, 5.0], 'Yb': [2.5, 3.0, 3.5, 4.1, 5.0], 'Lu': [2.2, 2.5, 3.04, 4.0, 5.0], 
'Hf': [1.975, 2.49, 3.25, 4.5], 'Ta': [1.85, 2.12, 2.625, 3.25, 4.5], 'W': [1.775, 1.99, 2.5, 3.25, 4.5], 
'Re': [1.775, 2.01, 2.5, 3.25, 4.25], 'Os': [1.8, 2.04, 2.5, 3.25, 4.5], 'Ir': [1.85, 2.125, 2.5, 3.25, 4.25], 
'Pt': [2.0, 2.275, 2.75, 3.75], 'Au': [2.1, 2.45, 3.0, 4.0], 'Hg': [2.225, 2.5, 3.04, 4.0, 5.0], 
'Tl': [2.21, 2.6, 3.11, 3.75, 4.75], 'Pb': [2.225, 2.5, 2.88, 3.625, 4.5], 'Bi': [2.225, 2.61, 3.125, 3.75, 4.75], 
'Po': [2.3, 2.72, 3.25, 3.875, 4.75], 'At': [2.375, 2.83, 3.5, 4.5], 'Rn': [2.8, 3.5, 4.17, 4.75, 5.5], 
'Fr': [2.85, 3.5, 4.43, 5.5], 'Ra': [3.15, 3.5, 4.25, 5.12, 6.0], 'Ac': [2.48, 3.1, 3.72, 4.25, 5.0], 
'Th': [2.25, 2.65, 3.25, 4.0, 5.0], 'Pa': [2.04, 2.3, 3.0, 3.75, 4.75], 'U': [1.89, 2.09, 2.75, 3.5, 4.5], 
'Np': [1.84, 2.05, 2.625, 3.375, 4.5], 'Pu': [1.81, 2.02, 2.5, 3.25, 4.25], 'Am': [1.81, 2.03, 2.5, 3.25, 4.25], 
'Cm': [1.83, 2.07, 2.5, 3.25, 4.25], 'Bk': [1.86, 2.12, 2.5, 3.0, 4.0], 'Cf': [1.89, 2.19, 2.625, 3.125, 4.0], 
'Es': [1.93, 2.29, 2.625, 3.125, 4.0]},
"trimer": {'S': [1.7, 2.2, 2.8], 'Pd': [2.2, 2.6, 3.2], 'Si': [1.9, 2.1, 2.6], 
'Te': [2.4, 2.8, 3.4], 'Sn': [2.3, 2.6, 3.1], 'Xe': [3.8, 4.3, 5.0], 'Mo': [1.8, 2.1, 2.7], 'In': [2.3, 2.8, 3.4], 
'Nb': [1.6, 1.9, 2.7], 'Ga': [2.3, 2.7, 3.4], 'Br': [2.1, 2.5, 3.0], 'Ir': [2.0, 2.3, 3.8], 'Be': [2.2, 2.7, 3.4], 
'W': [1.7, 1.9, 2.2], 'Mg': [2.7, 3.2, 3.9], 'Sb': [2.2, 2.7, 3.3], 'Re': [1.9, 2.2, 2.8], 'Ba': [3.2, 3.9, 4.7], 
'Rb': [3.9, 4.7, 5.5], 'Ag': [2.3, 2.7, 3.2], 'Hg': [2.7, 3.5, 4.3], 'Zn': [2.5, 3.2, 3.8], 'Cr': [1.5, 1.8, 2.3], 
'Os': [1.9, 2.2, 2.8], 'Na': [2.8, 3.4, 4.1], 'H': [0.7, 0.9, 1.3], 'Sc': [2.0, 2.5, 3.1], 'Zr': [2.1, 2.5, 3.1], 
'Se': [2.1, 2.3, 2.7], 'Al': [2.3, 2.8, 3.4], 'Rh': [2.0, 2.3, 2.7], 'Y': [2.4, 2.9, 3.6], 'B': [1.2, 1.5, 2.1], 
'Ca': [2.8, 3.6, 4.6], 'Fe': [1.6, 2.0, 2.9], 'Tc': [1.5, 1.8, 2.2], 'Cs': [4.3, 5.0, 5.8], 'Ne': [2.0, 2.7, 3.3], 
'C': [1.1, 1.4, 2.1], 'Ar': [2.8, 3.2, 3.7], 'He': [1.5, 2.0, 2.6], 'N': [0.9, 1.2, 1.6], 'Au': [2.3, 2.7, 3.2], 
'Pt': [2.2, 2.6, 3.2], 'F': [1.3, 1.6, 2.1], 'Ge': [1.9, 2.2, 2.8], 'Co': [2.0, 2.4, 2.9], 'Cl': [1.6, 1.8, 2.2], 
'Ti': [1.7, 2.2, 2.9], 'K': [3.0, 3.8, 4.6], 'V': [1.6, 1.9, 2.6], 'Cu': [2.0, 2.4, 3.0], 'Pb': [2.3, 2.7, 3.3], 
'O': [1.1, 1.4, 2.1], 'As': [2.0, 2.3, 2.7], 'Li': [1.9, 2.4, 3.3], 'Bi': [2.4, 2.9, 3.5], 'Ru': [1.8, 2.1, 2.7], 
'Sr': [3.5, 4.1, 4.7], 'Kr': [3.3, 4.0, 4.7], 'I': [2.4, 2.9, 3.5], 'Ta': [1.7, 2.0, 2.3], 'Mn': [1.5, 1.8, 2.5], 
'Tl': [2.4, 3.3, 4.3], 'Ni': [1.9, 2.3, 2.8], 'P': [1.7, 2.2, 2.8], 'Hf': [2.3, 2.8, 3.4], 'Cd': [2.7, 3.6, 4.5]}}

# only consider those parameters that are either directly or in-directly relevant with
# orbital generation tasks.
ABACUS_PARAMS = [
    'suffix', 'latname', 'stru_file', 'kpoint_file', 'pseudo_dir', 'orbital_dir', 
    'pseudo_rcut', 'pseudo_mesh', 'lmaxmax', 'dft_functional', 'xc_temperature', 
    'calculation', 'esolver_type', 'ntype', 'nspin', 'kspacing', 'min_dist_coef', 
    'nbands', 'symmetry', 'symmetry_prec', 'symmetry_autoclose', 'nelec', 'nelec_delta', 
    'out_mul', 'noncolin', 'lspinorb', 'kpar', 'bndpar', 'out_freq_elec', 'printe', 
    'mem_saver', 'diago_proc', 'nbspline', 'wannier_card', 'soc_lambda', 
    'cal_force', 'out_freq_ion', 'device', 'precision', 'ecutwfc', 'ecutrho', 
    'erf_ecut', 'erf_height', 'erf_sigma', 'fft_mode', 'pw_diag_nmax', 'diago_cg_prec', 
    'pw_diag_thr', 'scf_thr', 'scf_thr_type', 'init_wfc', 'init_chg', 'chg_extrap', 
    'out_chg', 'out_pot', 'out_wfc_pw', 'out_wfc_r', 'out_dos', 'out_band', 'out_proj_band', 
    'restart_save', 'restart_load', 'read_file_dir', 'nx', 'ny', 'nz', 'ndx', 'ndy', 'ndz', 
    'cell_factor', 'pw_seed', 'ks_solver', 'scf_nmax', 'relax_nmax', 'out_stru', 'force_thr', 
    'force_thr_ev', 'force_thr_ev2', 'relax_cg_thr', 'stress_thr', 
    'press1', 'press2', 'press3', 'relax_bfgs_w1', 'relax_bfgs_w2', 'relax_bfgs_rmax', 
    'relax_bfgs_rmin', 'relax_bfgs_init', 'cal_stress', 'fixed_axes', 'fixed_ibrav', 
    'fixed_atoms', 'relax_method', 'relax_new', 'relax_scale_force', 'out_level', 'out_dm', 
    'out_bandgap', 'use_paw', 'basis_type', 'gamma_only', 'search_radius', 'search_pbc', 
    'lcao_ecut', 'lcao_dk', 'lcao_dr', 'lcao_rmax', 'out_mat_hs', 'out_mat_hs2', 'out_mat_dh', 
    'out_mat_xc', 'out_interval', 'out_app_flag', 'out_mat_t', 'out_element_info', 'out_mat_r', 
    'out_wfc_lcao', 'bx', 'by', 'bz', 'smearing_method', 'smearing_sigma', 'mixing_type', 
    'mixing_beta', 'mixing_ndim', 'mixing_restart', 'mixing_gg0', 'mixing_beta_mag', 'mixing_gg0_mag', 
    'mixing_gg0_min', 'mixing_angle', 'mixing_tau', 'mixing_dftu', 'mixing_dmr', 'dos_emin_ev', 
    'dos_emax_ev', 'dos_edelta_ev', 'dos_scale', 'dos_sigma', 'dos_nche', 'md_type', 'md_thermostat', 
    'md_nstep', 'md_dt', 'md_tchain', 'md_tfirst', 'md_tlast', 'md_dumpfreq', 'md_restartfreq', 
    'md_seed', 'md_prec_level', 'ref_cell_factor', 'md_restart', 'lj_rcut', 'lj_epsilon', 
    'lj_sigma', 'pot_file', 'msst_direction', 'msst_vel', 'msst_vis', 'msst_tscale', 'msst_qmass', 
    'md_tfreq', 'md_damp', 'md_nraise', 'cal_syns', 'dmax', 'md_tolerance', 'md_pmode', 'md_pcouple', 
    'md_pchain', 'md_pfirst', 'md_plast', 'md_pfreq', 'dump_force', 'dump_vel', 'dump_virial', 
    'efield_flag', 'dip_cor_flag', 'efield_dir', 'efield_pos_max', 'efield_pos_dec', 'efield_amp', 
    'gate_flag', 'zgate', 'relax', 'block', 'block_down', 'block_up', 'block_height', 'out_alllog', 
    'nurse', 'colour', 't_in_h', 'vl_in_h', 'vnl_in_h', 'vh_in_h', 'vion_in_h', 'test_force', 
    'test_stress', 'test_skip_ewald', 'vdw_method', 'vdw_s6', 'vdw_s8', 'vdw_a1', 'vdw_a2', 'vdw_d', 
    'vdw_abc', 'vdw_C6_file', 'vdw_C6_unit', 'vdw_R0_file', 'vdw_R0_unit', 'vdw_cutoff_type', 
    'vdw_cutoff_radius', 'vdw_radius_unit', 'vdw_cn_thr', 'vdw_cn_thr_unit', 'vdw_cutoff_period', 
    'exx_hybrid_alpha', 'exx_hse_omega', 'exx_separate_loop', 'exx_hybrid_step', 'exx_mixing_beta', 
    'exx_lambda', 'exx_real_number', 'exx_pca_threshold', 'exx_c_threshold', 'exx_v_threshold', 
    'exx_dm_threshold', 'exx_cauchy_threshold', 'exx_c_grad_threshold', 'exx_v_grad_threshold', 
    'exx_cauchy_force_threshold', 'exx_cauchy_stress_threshold', 'exx_ccp_rmesh_times', 'exx_opt_orb_lmax', 
    'exx_opt_orb_ecut', 'exx_opt_orb_tolerence', 'td_force_dt', 'td_vext', 'td_vext_dire', 'out_dipole', 
    'out_efield', 'out_current', 'ocp', 'ocp_set', 'berry_phase', 'gdir', 
    'imp_sol', 'eb_k', 'tau', 'sigma_k', 'nc_k', 'dft_plus_u', 'yukawa_lambda', 'yukawa_potential',
    'omc', 'onsite_radius', 'hubbard_u', 'orbital_corr', 'bessel_nao_ecut', 'bessel_nao_tolerence', 
    'bessel_nao_rcut', 'bessel_nao_smooth', 'bessel_nao_sigma', 'bessel_descriptor_lmax', 
    'bessel_descriptor_ecut', 'bessel_descriptor_tolerence', 'bessel_descriptor_rcut', 
    'bessel_descriptor_smooth', 'bessel_descriptor_sigma', 'sc_mag_switch', 'decay_grad_switch', 
    'sc_thr', 'nsc', 'nsc_min', 'sc_scf_nmin', 'alpha_trial', 'sccut', 'sc_file',
    'out_mat_tk'] # for jy basis

class ABACUSIOTest(unittest.TestCase):

    def test_autoset(self):
        dftparam = autoset({'nspin': 2, 'basis_type': 'lcao',
                            'ks_solver': 'scalapack_gvx'})
        self.assertEqual(dftparam['nspin'], 2)
        self.assertEqual(dftparam['mixing_beta'], '0.4')
        self.assertEqual(dftparam['ks_solver'], 'scalapack_gvx')
        self.assertEqual(dftparam['out_mat_hs'], '1 12')
        self.assertEqual(dftparam['basis_type'], 'lcao')
        self.assertEqual(dftparam['out_wfc_lcao'], '1')

        with self.assertRaises(ValueError):
            _ = autoset({'non_existing_key': 1})
    
    def test_dftparam_to_text(self):
        dftparam = {'nspin': 2, 'basis_type': 'lcao',
                    'ks_solver': 'scalapack_gvx'}
        text = dftparam_to_text(dftparam)
        self.assertIn('nspin', text)
        self.assertIn('basis_type', text)
        self.assertIn('ks_solver', text)
        self.assertIn('lcao', text)
        self.assertIn('scalapack_gvx', text)
    
    def test_parse_abacus_dftparam(self):
        dftparam = {'nspin': '2', 'basis_type': 'lcao',
                    'ks_solver': 'scalapack_gvx'}
        text = dftparam_to_text(dftparam)
        dftparam2 = parse_abacus_dftparam(text)
        self.assertEqual(dftparam, dftparam2)

    def test_structure_to_text(self):
        
        # too long the bond length
        with self.assertRaises(ValueError):
            _ = structure_to_text(shape='dimer',
                                  element='Na',
                                  mass=1.,
                                  fpseudo='Na_ONCV_PBE-1.0.upf',
                                  lattice_constant=30.,
                                  bond_length=4.1,
                                  nspin=1,
                                  forb='Na_gga_14au_100Ry_4s2p1d.orb')
        
        # will not raise
        _ = structure_to_text(shape='dimer',
                              element='Na',
                              mass=1.,
                              fpseudo='Na_ONCV_PBE-1.0.upf',
                              lattice_constant=30.,
                              bond_length=4.1,
                              nspin=1,
                              forb='Na_gga_11au_100Ry_4s2p1d.orb')

if __name__ == '__main__':
    unittest.main()
