# built-in modules
import logging

# third-party modules
import numpy as np

# local modules
from SIAB.io.param import read, orb_link_geom
from SIAB.supercomputing.op import submit
from SIAB.abacus.api import build_abacus_jobs, job_done
from SIAB.io.convention import dft_folder, nzeta_string
from SIAB.orb.api import GetOrbCascadeInstance, DeriveCascadeInstance
from SIAB.orb.orb import Orbital
from SIAB.abacus.blscan import jobfilter
from SIAB.spillage.util import _spill_opt_param
from SIAB.driver.control import OrbgenAssert

def init(fn):
     '''
     initialize the ABACUS-ORBGEN workflow by reading the input file

     Parameters
     ----------
     fn: str
         input filename
     '''
     glbparams, dftparams, spillparams, compute, iop = read(fn)
     
     # if fit_basis is jy, then set basis_type in dftparams to lcao explicitly
     dftparams['basis_type'] = 'lcao' if spillparams.get('fit_basis', 'jy') == 'jy' else 'pw'
     
     # link the geometries
     for orb in spillparams['orbitals']:
          orb['geoms'] = orb_link_geom(orb['geoms'], spillparams['geoms'])
          
     return glbparams, dftparams, spillparams, compute, iop

def rundft(atomspecies,
           rcuts,
           dftparam,
           geoms,
           spillguess,
           compparam,
           **kwargs):
     '''
     run the ABACUS DFT calculations to generate the reference wavefunctions
     
     Parameters
     ----------
     atomspecies: list[dict]
          the atomspecies required to perform the DFT calculations
     rcuts: list[float]
          the cutoff radius of orbital to generate. this parameter will affect
          in two aspects: 1. in the ABACUS INPUT parameter, 2. the definition
          of the orbital to generate
     dftparam: dict
          other ABACUS INPUT parameters
     geoms: dict
          the geometries that ABACUS perform the DFT calculations on
     spillguess: str|None
          the initial guess of the spillage optimization, now it must be `atomic`
     compparam: dict
          the computational parameters, including `abacus_command`, `environment`,
          `mpi_command`
     ecutjy: float, optional
          the kinetic energy cutoff of the underlying jy, if not set, will use the
          value of ecutwfc in dftparam
     
     Returns
     -------
     jobs: list[str]
          the job names of the ABACUS calculations
     '''
     # placing the build_abacus_jobs ahead of the check of `abacus_command`,
     # supporting the case that only generate the jy orbitals, without running
     # all the dft calculations
     jobs = build_abacus_jobs(atomspecies=atomspecies, 
                              rcuts=rcuts, 
                              dftparams=dftparam, 
                              geoms=geoms, 
                              spill_guess=spillguess, 
                              **kwargs)
     
     # then run ABACUS
     abacus_command = compparam.get('abacus_command', 'abacus')
     if abacus_command is None:
          logging.info('abacus command is not found, workflow terminated.')
          exit()
     for job in jobs:
         if job_done(job):
             logging.info(f'{job} has been done, skip')
             continue
         _ = submit(job, 
                    compparam.get('environment', ''),
                    compparam.get('mpi_command', ''),
                    abacus_command)
     # to find these folders by name, call function
     # in SIAB.io.convention the dft_folder
     # dft_folder(elem, proto, pert, rcut = None)
     return jobs

def _spilltasks(elem, 
                rcuts, 
                scheme, 
                dft_root = '.',
                run_mode = 'jy',
                **kwargs):
     '''
     bind the spillage optimization tasks with the reference geometries
     
     Parameters
     ----------
     elem: str
         element symbol
     rcuts: list[float]
         the cutoff radius of orbital to generate
     scheme: list
          the scheme of how to generate the orbitals
     dft_root: str
          the root folder of the dft calculations
     run_mode: str
          the mode to execute spillage optimization, default is jy, also can be pw
     
     Generate
     --------
     rcut: float
          the cutoff radius of the orbitals
     orbitals: list[dict]
          the orbitals to optimize
     '''
     convert_ = {'nzeta': 'nzeta', 'nbands': 'nbnds', 
                 'checkpoint': 'iorb_frozen', 'geoms': 'folders'}
     
     logging.info('') # spacing with the previous log
     for rcut in rcuts:
          template = scheme.copy()
          orbitals = [{convert_.get(k, k): v for k, v in orb.items()} for orb in template]
          additional = {} if run_mode != 'jy' else {'rcut': rcut}
          for i, orb in enumerate(orbitals):
               geoms_orb = [{'elem': elem, 'proto': f['proto'], 'pert': pertmag} 
                            for f in orb['folders'] 
                            for pertmag in jobfilter(root=dft_root, 
                                                     elem=elem, 
                                                     proto=f['proto'],
                                                     pertkind='stretch', 
                                                     pertmags=f['pertmags'],
                                                     rcut=additional.get('rcut'),
                                                     n=kwargs.get('__iop_blscan_n__', 5), 
                                                     ethr=kwargs.get('__iop_blscan_de__', 1.5))]
               orb['folders'] = [dft_folder(**(geom|additional)) for geom in geoms_orb]
               OrbgenAssert(len(orb['folders']) > 0, f'No geometries linked for {i}-th orbital')
               logging.info(f'Reference geometries for {i}-th orbital with cutoff radius {rcut} au:')
               for f in orb['folders']:
                    logging.info(f)

          logging.info(f'Generate spillage optimization tasks for rcut = {rcut} au')
          yield rcut, orbitals

def minimize_spillage(elem, 
                      ecut, 
                      rcuts, 
                      primitive_type,  
                      scheme, 
                      dft_root = '.',
                      run_mode = 'jy',
                      outdir = '.',
                      **kwargs):
     '''
     Run the regular spillage optimization followed by an optional greedy algorithm 
     that starting from the orbital whose label `greedygrow` is True.
     Each time try to find the angular momentum of orbital that can reduce the 
     spillage most, and then add it to the basis set. The optimization will stop 
     when reach the nzeta_max.

     Parameters
     ----------
     elem: str
         element symbol
     ecut: float
          the kinetic energy cutoff of the underlying jy
     rcuts: list[float]
          the cutoff radius of orbital to generate
     primitive_type: str
          the type of jy, can be `reduced` or `normalized`
     scheme: list
          the scheme of how to generate the orbitals
     dft_root: str
          the root folder of the dft calculations
     run_mode: str
          the mode to execute spillage optimization, default is jy, also can be pw
     outdir: str
          the output directory, default is the current directory
     kwargs: dict
          additional parameters, including `max_steps`, `verbose`, `ftol`, `gtol`, `nthreads_rcut`
     '''
     optimizer, options = _spill_opt_param(kwargs)
     for rcut, tasks in _spilltasks(elem, rcuts, scheme, dft_root, 
                                    run_mode, **kwargs.get('iop', {})):
          # set the default initial guess generation information
          initializer = {'model': kwargs.get('spill_guess', 'atomic'),
                         'model_kwargs': {}}
          if initializer['model'] == 'atomic':
             temp = {} if run_mode != 'jy' else {'rcut': rcut}
             initializer['model_kwargs'] = {'jobdir': dft_folder(elem, 'monomer', 0, **temp)}

          # the following parameters are shared by all the cascades
          shared_cascade_param = {'elem': elem, 'rcut': rcut, 'ecut': ecut,
                                  'primitive_type': primitive_type}
          # the first cascade is the minimal basis
          cascade_ = GetOrbCascadeInstance(**shared_cascade_param,
                                           initializer=initializer,
                                           orbparam=tasks,
                                           mode=run_mode,
                                           optimizer=optimizer)

          cascade_, spill_ = cascade_.opt(diagnosis=True,
                                          options=options,
                                          nthreads=kwargs.get('nthreads_rcut', 1))

          logging.info(f'Normal spillage optimization completed.')
          # ======================================================================= #
          # 
          for i in range(len(tasks)):
               if not tasks[i].get('greedygrow', False):
                    continue
               logging.warning('Greedy algorithm is enabled for this task, '
                               'will try to add more orbitals to the basis set. '
                               'If there are any two orbitals with the same number'
                               'of zeta functions, overwriting will happen.')

               cascade_, _ = greedygrow_cascade(
                    cascade=cascade_,
                    growpos=i,
                    nzeta_min=tasks[i]['nzeta'],
                    nzeta_max=tasks[i].get('nzeta_max', tasks[i]['nzeta']),
                    shared_orb_param=tasks[i].copy(),
                    shared_cascade_param=shared_cascade_param,
                    spill_init=spill_[i],
                    minimization=options,
                    **kwargs)
          # ======================================================================= #
          #
          logging.info('') # spacing with the previous log
          logging.info(f'Printing the orbitals (param/orb/png) to files in {outdir}')
          cascade_.to_file(outdir=outdir)

def greedygrow_cascade(cascade,
                       growpos,
                       nzeta_min,
                       nzeta_max,
                       shared_orb_param,
                       shared_cascade_param,
                       spill_init,
                       minimization,
                       **kwargs):
     '''
     employ the greedy algorithm to organize radial functions based on one
     starting point
     
     Parameters
     ----------
     nzeta_min: list[int]
          the number of zeta functions for each angular momentum of the
          starting point, e.g. [1, 1, 0, 0] means 1s1p. If the `nzeta_max`
          is given as [2, 2, 1, 1], then the greedy algorithm will try to
          add one more orbital to the basis set with angular momentum
          from 0 to 3, then find the one that can reduce the spillage the
          most efficiently, and then add it to the basis set. The greedy
          algorithm will stop when all the angular momentums have been
          exhausted, or the `nzeta_max` is reached.
     nzeta_max: list[int]
          the upper bound of the number of zeta functions for each angular
          momentum. If not given, it will be the same as `nzeta_min`, then
          the function will directly return
     '''
     nzeta_pool = [b - a for a, b in zip(nzeta_min, nzeta_max)]
     OrbgenAssert(all([nz >= 0 for nz in nzeta_pool]),
                    '`nzeta_max` should be larger than `nzeta_min`')
     spill_ = spill_init
     # trajectories
     trajectories = {'nzeta': [nzeta_min], 'Spillage': [spill_]}
     while any([nz > 0 for nz in nzeta_pool]): # if the pool is still not expired
          logging.info(f'Residual available numbers of zeta functions in pool: {nzeta_pool}')
          l_pool = [l for l, nz in enumerate(nzeta_pool) if nz > 0]
          
          # build several trial cascades, each with one additional orbital
          nzeta_new = [[nz + int(l == l_) 
                        for l, nz in enumerate(trajectories['nzeta'][-1])] 
                       for l_ in l_pool] # will used to print on screen later
          cascades_trial = [DeriveCascadeInstance(
                              cascade=cascade,
                              orbparam=[shared_orb_param.copy() | {
                                   'nzeta': nz,
                                   'iorb_frozen': growpos}],
                              **shared_cascade_param) 
                            for nz in nzeta_new]
          
          # run the spillage optimization
          spill = np.array([cascade.opt(diagnosis=True,
                                        options=minimization,
                                        nthreads=kwargs.get('nthreads_rcut', 1))[1][-1]
                    for cascade in cascades_trial]).flatten().tolist()

          # calculate the delta-spillage for those newly added orbitals
          # note: the dspill is divided by (2*l+1) to account for the
          # computational cost
          dspill = [(s - spill_)/(2*l+1) for l, s in zip(l_pool, spill)]
          jmin = np.argmin(dspill) # we use `j` to index the `l_pool`
          
          # report
          logging.info('Greedy algorithm takes one step.')
          logging.info(f'{"l":>2}    {"Spillage":>14} {"DSpill/(2l+1)":>15}')
          logging.info('-'*(2+4+14+1+15))
          for j, (l, s, ds) in enumerate(zip(l_pool, spill, dspill)):
               temp = f'{l:>2} -> {s:>14.8e} {ds:>+15.8e}'
               if j == jmin:
                    temp += ' <- selected'
               logging.info(temp)

          # convergence check
          # if all the delta-spillage are positive
          if all(ds > 0 for ds in dspill):
               i = np.argmin(spill)
               warn  = f'More zeta functions cannot decrease the Spillage Greedy algorithm stops.'
               warn += f' The orbital with the lowest Spillage is {nzeta_new[i]},'
               warn += f' corresponding Spillage value is decreased to {spill[i]:<14.8e}'
               logging.warning(warn)
               break
          
          # otherwise, we update the loop control variables
          cascade, spill_ = cascades_trial[jmin].copy(), spill[jmin]
          trajectories['nzeta'].append(nzeta_new[jmin])
          trajectories['Spillage'].append(spill_)
          growpos = cascade.get_num_orbitals() - 1 # then grow from the newly added orbital
          
          # report the details of spillage decrease trajectory
          temp = f'Greedy algorithm selects l = {l_pool[jmin]}, with nzeta = {nzeta_new[jmin]}'
          logging.info(temp)
          logging.info(f'Greedy algorithm yields Spillage decrease trajectory:')
          logging.info(f'# nrad: total number of radial functions')
          logging.info(f'# nbasis: total number of basis functions (computional cost)')
          logging.info(f'# Spillage: the spillage value of the basis set')
          logging.info(f'{"nrad":>6} {"nbasis":>6} {"Spillage":>14}')
          logging.info('-'*(6+1+6+1+14))
          for nz, s in zip(trajectories['nzeta'], trajectories['Spillage']):
               temp = f'{np.sum(nz):>6} {Orbital.nloc(nz):>6} {s:>14.8e} # {nzeta_string(nz)}'
               logging.info(temp)
          logging.info('')
          
          # continue the greedy algorithm
          nzeta_pool = [nz - int(l == l_pool[jmin]) for l, nz in enumerate(nzeta_pool)]

     return cascade, trajectories
