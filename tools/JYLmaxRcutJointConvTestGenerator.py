'''
Concept
-------
Different structure's Lmax-Rcut Joint Convergence test
can be fully parallelized. This generator will generate
the Lmax-Rcut Joint Convergence test for all bond lengths
provided for one element. Each bond length will have its
own folder. Then the abacustest Python package will be
used to submit the jobs to the Bohrium platform.

For all bond lengths, they can share the same set of 
jY orbitals. But due to there is no signal to know when
the job files have been uploaded to the Bohrium platform,
it is impossible to do like, generate for once, move
to the bond length corresponding folder one by one,
submit the job and move the jY orbitals to the next bond
length folder. Instead, will generate jY orbitals for
once, then copy to all bond length folders.
'''
import re
import os
import json
import shutil
import time
import logging

# the bond length data for elements are stored here
DIMER_BOND_LENGTH_7POINTS = {
    "H":  [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    "Li": [2.45, 2.50, 2.55, 2.60, 2.65, 2.70, 2.75],
    "Be": [2.30, 2.35, 2.40, 2.45, 2.50, 2.55, 2.60],
    "B":  [1.45, 1.50, 1.55, 1.60, 1.65, 1.70, 1.75],
    "C":  [1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40],
    "N":  [0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25],
    "O":  [1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35],
    "F":  [1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55],
    "Ne": [2.95, 3.00, 3.05, 3.10, 3.15, 3.20, 3.25],
    "Na": [2.95, 3.00, 3.05, 3.10, 3.15, 3.20, 3.25],
    "Mg": [3.75, 3.80, 3.85, 3.90, 3.95, 4.00, 4.05],
    "Al": [2.55, 2.60, 2.65, 2.70, 2.75, 2.80, 2.85],
    "Si": [2.10, 2.15, 2.20, 2.25, 2.30, 2.35, 2.40],
    "P":  [1.75, 1.80, 1.85, 1.90, 1.95, 2.00, 2.05],
    "S":  [1.75, 1.80, 1.85, 1.90, 1.95, 2.00, 2.05],
    "Cl": [1.85, 1.90, 1.95, 2.00, 2.05, 2.10, 2.15],
    "Ar": [3.60, 3.65, 3.70, 3.75, 3.80, 3.85, 3.90]
}

DIMER_BOND_LENGTH_5POINTS = {'H': [0.6, 0.75, 0.9, 1.2, 1.5], 'He': [1.25, 1.75, 2.4, 3.25], 
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
'Es': [1.93, 2.29, 2.625, 3.125, 4.0]}

def _convert_to_int_if_possible(s, thr = 1e-6):
    '''
    convert a float number to int if possible
    '''
    if abs(s - round(s)) < thr:
        return int(round(s))
    return s

def _parse_orb_comp(orb):
    '''
    parse the orb components information in pattern like '21s20p20d19f19g18h'
    '''
    m = re.match(r'((\d+)([spdfghiklm]))+', orb)
    if not m:
        raise ValueError(f'Invalid orb component: {orb}')
    spectrum = 'spdfghiklm'
    lmax = 0
    for s in spectrum[::-1]:
        if s in orb:
            lmax = spectrum.index(s) + 1
            break
    temp = dict([(m.group(2), int(m.group(1))) for m in re.finditer(r'(\d+)([spdfghiklm])', orb)])
    return [temp.get(spectrum[l], 0) for l in range(lmax)]

def _parse_forb(fn):
    '''
    parse the information of an ABACUS orbital file by its name
    '''
    
    fn = os.path.basename(fn) if '/' in fn else fn
    # abacus orbital files' names in format N_gga_7au_100Ry_21s20p20d19f19g18h.orb
    pat = r'^([A-Z][a-z]?)_gga_(\d+)au_(\d+(\.\d+)?)Ry_(\w+)\.orb$'
    m = re.match(pat, fn)
    if not m:
        raise ValueError(f'Invalid orbital file name: {fn}')
    elem, rcut, ecut, _, orbs = m.groups()
    return dict([('elem', elem), 
                 ('rcut', _convert_to_int_if_possible(float(rcut))), 
                 ('ecut', _convert_to_int_if_possible(float(ecut))), 
                 ('nzeta', _parse_orb_comp(orbs))])

def _change_elem_in_orb(fn, elem):
    '''
    change the element symbol in orbital file
    '''
    fn_pat = r'^([A-Z][a-z]?)_gga_(\d+)au_(\d+(\.\d+)?)Ry_(\w+)\.orb$' 
    elem_pat = r'(Element)(\s+)([A-Z][a-z]?)'

    if not os.path.isfile(fn):
        raise FileNotFoundError(f"file {fn} not found.")
    
    if not re.match(fn_pat, os.path.basename(fn)):
        raise ValueError(f"file {fn} does not match the pattern.")

    fn_new = re.sub(fn_pat, f'{elem}_gga_\\2au_\\3Ry_\\5.orb', os.path.basename(fn))

    with open(fn) as f:
        lines = f.readlines()

    with open(fn_new, 'w') as f:
        for line in lines:
            if re.match(elem_pat, line):
                f.write(re.sub(elem_pat, f'\\1\\2{elem}', line))
            else:
                f.write(line)

    return fn_new

def _build_orb_from_exists(target, outdir, elem, ecutjy, rcut, lmax):
    '''
    because the jY orbital is shared by all elements, because it is
    pseudopotential-irrelevant.

    Parameters
    ----------
    target: str
        the folder in which all proto-orbitals are stored.
    
    outdir: str
        the folder in which the new orbitals will be stored.

    elem: str
        the element name.
    
    ecutjy: float
        the cutoff kinetic energy of spherical wave basis
    
    rcut: float
        the cutoff radius of spherical wave basis
    
    lmax: int
        the maximum angular momentum quantum number
    '''
    forb = []
    fn_pat = r'^([A-Z][a-z]?)_gga_(\d+)au_(\d+(\.\d+)?)Ry_(\w+)\.orb$'
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(target):
        m = re.match(fn_pat, f)
        if m:
            _, rcut_, ecut_, nzeta_ = _parse_forb(f).values()
            lmax_ = len(nzeta_) - 1
            if (rcut is None or abs(float(rcut_) - rcut) <= 1e-6) and \
               (ecutjy is None or abs(float(ecut_) - ecutjy) <= 1e-6) and \
               (int(lmax_) == lmax or lmax is None): # then it is the one that can be used
                forb_new = _change_elem_in_orb(os.path.join(target, f), elem)
                os.rename(forb_new, os.path.join(outdir, os.path.basename(forb_new)))
                forb.append(forb_new)
        else:
            print(f'file {f} does not match the pattern.')
    return forb

def _jygen(elem, fpsp, ecutjy, rcut, lmax, outdir, orbgen):
    '''
    generate the jY orbital for the element with the given parameters
    '''
    rcut = [rcut] if isinstance(rcut, (int, float)) else rcut
    rcut = [_convert_to_int_if_possible(r) for r in rcut]
    orb_in = {'environment': '', 'mpi_command':'', 'abacus_command': None, 
              'element': elem, 'ecutwfc': ecutjy,
              'pseudo_dir': fpsp,
              'fit_basis': 'jy', 'bessel_nao_rcut': rcut, 'primitive_type': 'reduced',
              'optimizer': 'torch.swats',
              'geoms': [
                  {'proto': 'dimer', 'pertkind': 'stretch', 'pertmags': [1.00], 
                   'nbands': 10, 'lmaxmax': lmax}
              ],
              'orbitals': [
                  {'nzeta': [1, 1, 0], 'geoms': [0], 'nbands': ['occ'], 'checkpoint': None}
              ]}
    
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, 'jygen.json'), 'w') as f:
        json.dump(orb_in, f, indent=4)
    cwd = os.getcwd()
    os.chdir(outdir)
    os.system(f'{orbgen} -i jygen.json') # will generate a folder named primitive_jy
    forbs = [f for f in os.listdir('primitive_jy') if f.endswith('.orb')]
    os.system(f'rm -rf {elem}-*')
    os.chdir(cwd)
    for f in forbs:
        os.rename(os.path.join(outdir, 'primitive_jy', f), os.path.join(outdir, f))
    os.system(f'rm -rf {os.path.join(outdir, "primitive_jy")}')
    return outdir, [os.path.basename(f) for f in forbs]

def _write_driver_inp(outdir, abacus, elem, bl, l, r, fpsp, nscf, celldm, orbdir, ecutgrid):
    '''
    write the driver.inp file for the LmaxConvergenceTestDriver.py
    '''
    c = {
        'abacus_command': abacus, 
        'lmin': l[0], 'lmax': l[1], 'dl': l[2],
        'rmin': r[0], 'rmax': r[1], 'dr': r[2],
        'elem': elem,
        'bl': bl,
        'fpsp': fpsp,
        'celldm': celldm,
        'nscf_energy': nscf,
        'orbital_dir': orbdir,
        'ecutwfc': ecutgrid
    }
    with open(os.path.join(outdir, 'driver.json'), 'w') as f:
        json.dump(c, f, indent=4)
    
    return os.path.join(outdir, 'driver.json')

def main(elem, 
         ecutjy, 
         r, 
         l, 
         ecutgrid,
         celldm, 
         nscf,
         fpsp,
         orbgen,
         abacustest = '__local__',
         fdriver = 'JYLmaxRcutJointConvTestDriver.py'):
    '''
    stages are:
    1. generate the jY orbitals
    2. create the folders for each bond length
    3. write the driver.inp for each bond length
    4. copy the jY orbitals to each bond length folder
    5. submit the jobs to the Bohrium platform. The job is not simply the abacus
       job, but to run JYLmaxRcutConvergenceTest.py on the Bohrium platform, in
       one abacus-installed Bohrium image. Seems the abacus v3.8.1 is a perfect
       one.

    manually, after all jobs submitted, the jY orbitals can
    be deleted.
    '''
    # we perform the sanity check elsewhere
    # 0. first get bond length data. If not supported, will raise and return at the beginning,
    # will not waste time on generating jY for this unsupported element.
    bls = []
    try:
        bls = DIMER_BOND_LENGTH_7POINTS.get(elem)
        bls = DIMER_BOND_LENGTH_5POINTS[elem] if bls is None else bls
    except KeyError:
        logging.error(f'Element {elem} is not supported.')
        return None
    
    # 1. generate the jY orbitals
    orbdir = f'{elem}-jy-ecutjy{ecutjy}Ry'
    for lmax in l:
        logging.info(f'Generating jY orbitals for {elem} with ecutjy={ecutjy}Ry, lmax={lmax}')
        _, _ = _jygen(elem, fpsp, ecutjy, r, lmax, orbdir, orbgen)
    # copy the psp into the orbital folder
    logging.info(f'Copying the pseudopotential {fpsp} to the orbital folder.')
    shutil.copy(fpsp, os.path.join(orbdir, os.path.basename(fpsp)))

    # 2. create the folders for each bond length
    elemdir = f'JYLmaxRcutJointConvTest-{elem}'
    logging.info(f'Creating the folders for each bond length in {elemdir}')

    celldm_lo = max(bls) + 2*max(max(r), 10) / 1.8897259886 # 1 Bohr = 0.52917721067 Angstrom
    celldm_hi = round(celldm_lo * 1.1) + 1

    if celldm_lo > celldm:
        errmsg = f'celldm is too small for the bond length {max(bls)} and rcut {max(r)}'
        logging.error(errmsg)
        raise ValueError(errmsg)
    if celldm_hi < celldm:
        logging.warning(f'celldm is reset to {celldm_hi} to save computational resources.')
        celldm = celldm_hi

    for bl in bls:
        bldir = os.path.join(elemdir, f'{elem}-{bl}')
        os.makedirs(bldir, exist_ok=True)
        
        # 3. write the driver.inp for each bond length
        logging.info(f'Writing the driver.inp for {elem} with bond length {bl}')
        orbdir_ = os.path.basename(orbdir)
        fpsp_ = os.path.join(orbdir_, os.path.basename(fpsp))
        fn = _write_driver_inp(outdir=bldir, 
                               abacus='OMP_NUM_THREADS=1 mpirun -np 16 abacus | tee abacus.out', 
                               elem=elem, 
                               bl=bl, 
                               l=[l[0], l[-1], 1],
                               r=[r[0], r[-1], 1],
                               nscf=nscf,
                               fpsp=fpsp_, 
                               celldm=celldm,
                               orbdir=orbdir_, # indicates the pseudopotential saves into this folder
                               ecutgrid=ecutgrid)
        
        # 4. copy the jY orbitals to each bond length folder
        shutil.copytree(orbdir, os.path.join(bldir, os.path.basename(orbdir)))
        # copy the driver into the bond length folder
        shutil.copy(fdriver, os.path.join(bldir, os.path.basename(fdriver)))

        # continue # for test

        # 5. submit the jobs to the Bohrium platform
        #    note: do not run abacus, but run with `python3 JYLmaxRcutJointConvTestDriver.py -i driver.inp`
        #    in Bohrium image `registry.dp.tech/dptech/abacus:3.8.1`
        if abacustest == '__local__':
            logging.info(f'Not submitting the jobs for {elem} with bond length {bl} because abacustest is `__local__`')
            continue

        logging.info(f'Submitting the jobs for {elem} with bond length {bl}')
        command =  f'{abacustest}'
        command += f' -i "registry.dp.tech/dptech/abacus:3.8.1"'
        command += f' -f {bldir}'
        command += f' -c "python3 {os.path.basename(fdriver)} -i {os.path.basename(fn)}"'
        logging.info(f'run abacustest with command: {command}')
        os.system(command)

    # remove the temporary jY orbitals
    shutil.rmtree(orbdir)
    logging.info(f'Temporary jY orbitals folder {orbdir} has been removed.')

    return elemdir

if __name__ == '__main__':
    '''this is only for running interactively'''
    prefix = 'JYLmaxRcutJointConvTest'
    fn = f'{prefix}@{time.strftime("%Y%m%d-%H%M%S")}.log'
    logging.basicConfig(filename=fn, level=logging.INFO)
    
    elem = 'Ru'
    #pseudo_dir = '/root/abacus-develop/pseudopotentials/sg15_oncv_upf_2020-02-06'
    #pseudo_dir = '/root/abacus-develop/pseudopotentials/nc-fr-04_pbe_standard/'
    #pseudo_dir = '/root/abacus-develop/pseudopotentials/NCPP-PD04-PBE'
    pseudo_dir = '/root/documents/simulation/abacus/apns-the-best-pp-orb/apns-psp-release-13th-Feb-2025/'
    ecutjy = 60
    ecutgrid = 100
    nscf = True # perform NSCF calculation instead of SCF
    
    # note:
    # you should set the ecutgrid larger than the conv_val 
    # of BOTH Vloc and grid integration on jY
    # for the former, plz refer to: 
    # https://kirk0830.github.io/ABACUS-Pseudopot-Nao-Square/pseudopotential/pseudopotential.html
    # for the latter, always set ecutgrid >= ecutjy + 10

    fpsp = os.path.join(pseudo_dir, f'{elem}_ONCV_PBE_FR-1.0.upf')

    jobdir = main(
        elem=elem,
        ecutjy=ecutjy,
        r=[6, 7, 8, 9, 10, 11, 12],
        l=[2, 3, 4],
        ecutgrid=ecutgrid,
        celldm=30,
        nscf=nscf,
        fpsp=fpsp,
        orbgen='orbgen',
    )
    print(f'Submitted test will be in {jobdir}')
    logging.info(f'Submitted test will be in {jobdir}')

    logging.shutdown() # close the log file
    print(f'Log file is saved in {fn}')
