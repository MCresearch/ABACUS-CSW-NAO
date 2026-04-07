'''
Concept
-------
The script is for generating all required files for spherical wave basis
kinetic energy cutoff convergence test light-weightingly as much as possible,
without calling the APNS, that is too heavy for this purpose

Technically, the convergence test can be carried out for one single atom,
because the kinetic energy cutoff is to including all spherical waves
whose kinetic energy is less than the cutoff. If the spherical wave can
represent will a wavefunction of an atom, it will also be valid for the
molecule. 
'''
import re
import os
import json
import shutil
import time
import logging

def _convert_to_int_if_possible(s, thr = 1e-6):
    '''
    convert a float number to int if possible
    '''
    if abs(s - round(s)) < thr:
        return int(round(s))
    return s

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

def _write_driver_inp(outdir, abacus, elem, fpsp, celldm, orbdir, ecutjy, ecutgrid):
    '''
    write the driver.json file for JYEkinConvTestDriver.py
    '''
    c = {
        'abacus_command': abacus,
        'elem': elem,
        'fpsp': fpsp,
        'celldm': celldm,
        'orbital_dir': orbdir,
        'ecutjy': ecutjy,
        'ecutwfc': ecutgrid
    }
    with open(os.path.join(outdir, 'driver.json'), 'w') as f:
        json.dump(c, f, indent=4)

    return os.path.join(outdir, 'driver.json')

def main(elem, 
         ecutjy,
         ecutgrid,
         celldm,
         fpsp,
         orbgen,
         lmax = 3,
         rcut = 10,
         abacustest = '__local__',
         fdriver = 'JYEkinConvTestDriver.py'):
    '''
    stages are:
    1. generate the jY orbitals
    2. write the driver.json file
    '''
    # 1. generate the jY orbitals
    orbdir = f'{elem}-jy-lmax{lmax}-{rcut}au'
    for e in ecutjy:
        logging.info(f'Generating jY orbitals for {elem} with ecutjy = {e} Ry, lmax = {lmax}, rcut = {rcut} au')
        _, _ = _jygen(elem, fpsp, e, rcut, lmax, orbdir, orbgen)
    # copy the psp into the orbital folder
    logging.info(f'Copying the pseudopotential {fpsp} to the orbital folder')
    shutil.copy(fpsp, os.path.join(orbdir, os.path.basename(fpsp)))

    # create the root folder of series of jobs
    elemdir = f'JYEkinConvTest-{elem}'
    logging.info(f'Creating the root folder {elemdir}')
    os.makedirs(elemdir, exist_ok=True)

    # 2. create the job folder for each batch of ecutjy, here we have only one
    jobdir = os.path.join(elemdir, 'ecutjy' + '-'.join([str(e) for e in ecutjy]))
    os.makedirs(jobdir, exist_ok=True)

    # 3. write the driver.json file
    orbdir_ = os.path.basename(orbdir)
    fn = _write_driver_inp(outdir=jobdir, 
                           abacus='OMP_NUM_THREADS=1 mpirun -np 16 abacus | tee abacus.out', 
                           elem=elem, 
                           fpsp=os.path.join(orbdir_, os.path.basename(fpsp)), 
                           celldm=celldm, 
                           orbdir=orbdir_, 
                           ecutjy=ecutjy, 
                           ecutgrid=ecutgrid)
    logging.info(f'Writing the driver.json file for {fdriver}')

    # 4. copy the jY orbitals to the job folder
    shutil.copytree(orbdir, os.path.join(jobdir, orbdir))
    # copy the driver into the job folder
    shutil.copy(fdriver, os.path.join(jobdir, fdriver))

    # remove the temporary jY orbitals
    shutil.rmtree(orbdir)
    logging.info(f'Temporary jY orbitals folder {orbdir} has been removed.')

    # 5. submit the jobs
    if abacustest == '__local__':
        logging.info(f'Not submitting the jobs for {elem} kinetic energy cutoff convergence test\
 because abacustest is `__local__`')
        return
    
    logging.info(f'Submitting the jobs for {elem} kinetic energy cutoff convergence test')
    command =  f'{abacustest}'
    command += f' -i "registry.dp.tech/dptech/abacus:3.8.1"'
    command += f' -f {elemdir}'
    command += f' -c "python3 {os.path.basename(fdriver)} -i {os.path.basename(fn)}"'
    logging.info(f'run abacustest with command: {command}')
    os.system(command)

    return elemdir

if __name__ == '__main__':

    '''this is only for running interactively'''
    prefix = 'JYEkinConvTest'
    fn = f'{prefix}@{time.strftime("%Y%m%d-%H%M%S")}.log'
    logging.basicConfig(filename=fn, level=logging.INFO)

    elem = 'Si'
    fpsp = '/path/to/your/psp/file'
    ecutjy = [10, 20, 40, 60, 80, 100, 150, 200]
    ecutgrid = 250

    jobdir = main(elem=elem,
                  ecutjy=ecutjy,
                  ecutgrid=ecutgrid,
                  celldm=30,
                  fpsp=fpsp,
                  orbgen='orbgen',
                  lmax=3, # for any present atom, there is no g orbital, so lmax=3 is enough
                  rcut=10)

    logging.shutdown() # close the log file
    print(f'Log file is saved in {fn}')
