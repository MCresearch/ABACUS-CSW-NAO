'''this tool is for inferring the feasible numbers of zeta functions
based on the wavefunction analysis techniques. For details, please
see the code implmentation in SIAB/spillage/lcao_wfc_analysis.py'''

import re
import json
import uuid
import tqdm
import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
plt.rcParams['text.usetex'] = True

from SIAB.spillage.datparse import (
    read_triu,
    read_istate_info,
    read_wfc_lcao_txt,
    read_input_script,
    read_running_scf_log
)
from SIAB.spillage.lcao_wfc_analysis import (
    typewise_wavefunction_singular_value_decomposition as twsvd,
    atomwise_maximum_wavefunction_singular_value_decomposition as amwsvd
)

def is_abacus_dir(jobdir: str) -> bool:
    '''check if the jobdir is a valid abacus job directory'''
    jobdir = Path(jobdir)
    # not exist
    if not jobdir.exists():
        return False
    # not a directory
    if not jobdir.is_dir():
        return False
    # no INPUT file
    if not (jobdir / 'INPUT').exists():
        return False
    # check with more details...
    dftparam = read_input_script(jobdir / 'INPUT')
    # no STRU file
    if not (jobdir / dftparam.get('stru_file', 'STRU')):
        return False
    # not finished
    outdir = jobdir / f'OUT.{dftparam.get("suffix", "ABACUS")}'
    if not outdir.exists():
        return False
    # no running_scf.log
    if not (outdir / f'running_{dftparam.get("calculation", "scf")}.log').exists():
        return False
    return True

def is_reference_dir(jobdir: str | Path) -> bool:
    '''check if it is the reference case'''
    # have the basename
    jobdir = Path(jobdir).resolve()
    return bool(re.match(r'[A-Z][a-z]?-(dimer|trimer)-\d+\.\d+(-\d+(.\d+)au)?', jobdir.name))

def parse_bandexpr(bandexpr: str) -> List[int]:
    '''parse the band expression, supported: `1:10` and `1:10:2`. 
    `1:10` means the band range considered would be firstly from 0-1, then
    perform the second run of analysis from 0-2, then 0-3, ..., 0-10.
    `1:10:2` means the 0-1, then 0-3, 0-5, ...'''
    assert re.match(r'\d+:\d+(:\d+)?', bandexpr)
    words = list(map(int, bandexpr.split(':')))
    assert len(words) in [2, 3]
    if len(words) == 2:
        words.append(1)

    # assert on the content of words
    assert words[0] > 0        # the starting point
    assert words[1] > words[0] # the ending point
    assert words[2] > 0        # the step size

    return np.arange(words[0], words[1]+1, words[2]).tolist()

def main(jobdir: str,
         bandexpr: str,
         method: str = 'twsvd',
         threshold: Optional[float] = None):
    ''''''
    jobdir = Path(jobdir)
    # if it is not directly the ABACUS jobdir, then assume there would be
    # folders in it
    jobdir = [jobdir] if is_abacus_dir(jobdir) else \
        [p for p in jobdir.iterdir() 
         if is_abacus_dir(p) and is_reference_dir(p)]

    fanalyz = {'twsvd': twsvd, 'amwsvd': amwsvd}[method]
    
    # if there are multiple folders, use the one for all
    nbands = parse_bandexpr(bandexpr)

    for geom in jobdir:
        print('Geometry analyzing:', geom.name)
        # dftparam = read_input_script(geom / 'INPUT')
        outdir = geom / f'OUT.ABACUS'
        
        wfc = read_wfc_lcao_txt(outdir / 'WFC_NAO_GAMMA1.txt')[0]
        print('    >> Wavefunction has been read into memory.')
        S = read_triu(outdir / 'data-0-S')
        print('    >> Overlap matrix has been read into memory.')
        dat = read_running_scf_log(outdir / 'running_scf.log')
        print('    >> SCF data has been read into memory.')
        
        # analyze
        print('    >> Start calculation')
        # Changelog: tqdm is good, help to know where we are :)
        out = [fanalyz(wfc, 
                       S, 
                       nb, 
                       dat['natom'], 
                       dat['nzeta'], 
                       threshold=threshold)
               for nb in tqdm.tqdm(nbands,desc='    >> Band analysis')]
        print('    >> Done')
        # transpose to the tuple so that 
        # singular values: accessed by index-0; [iband][it][l][ibas]
        # contraction:     accessed by index-1; [iband][it][l]
        # loss (not-used): accessed by index-2; [iband]
        out = tuple(zip(*out))

        print('    >> Exporting data to numpy (.npy) and JSON (.json) file')
        # export the singular values data
        datastamp = str(uuid.uuid4().hex)[:8]
        for it in range(len(dat['natom'])):
            singular_values = [out[0][ib][it] for ib in range(len(nbands))] # (ib, l, ibas) -> float
            # convert to list[float] to avoid the not JSON-serializable error
            singular_values = [[np.array(sv_bl).tolist() for sv_bl in sv_b] for sv_b in singular_values]
            with open(geom / f'JYWFCAnalysis{datastamp}-Atomtyp{it}.json', 'w') as f:
                json.dump(dict(zip(nbands, singular_values)), f, indent=4)
        print('    >> Done')
        
        # plot
        for it in tqdm.tqdm(range(len(dat['natom'])), 
                            desc='    >> Figure plotting'): # for each type
            contraction = np.array([out[1][ib][it] for ib in range(len(nbands))]) # (nbands, l) -> int

            assert contraction.ndim == 2
            assert contraction.shape == (len(nbands), len(dat['nzeta'][it]))

            fig, ax = plt.subplots()
            for l, cl in enumerate(contraction.T): # for each l, we plot a different line
                ax.plot(nbands, cl, 'o-', label=f'l={l}')
            
            # set styles - handle too many bands by reducing xtick density
            if len(nbands) > 10:
                # For large number of bands, show every nth tick
                step = max(1, len(nbands) // 10)  # Show roughly 10 ticks max
                tick_positions = nbands[::step]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_positions)
            else:
                # For small number of bands, show all normally
                ax.set_xticks(nbands)
            ax.set_yticks(np.arange(0, contraction.max()+1))
            ax.set_xlabel('Number of bands included')
            ax.set_ylabel('$n(\zeta)$')
            ax.set_title(f'Contraction for type {it}')

            ax.legend()
            # tight layout
            fig.tight_layout()
            # save figure
            fig.savefig(geom / f'JYWFCAnalysis{datastamp}-Atomtyp{it}.png', dpi=300)
            plt.close(fig)
        # done
        print('Done')

if __name__ == '__main__':

    myparser = argparse.ArgumentParser(description='JYWFCAnalyzer')
    myparser.add_argument('-i', '--inputfolder', 
                          required=True,
                          help='the input folder containing the ABACUS job directories')
    myparser.add_argument('-n', '--nbands', 
                          required=True,
                          help='the number of bands to be considered')
    myparser.add_argument('-m', '--method', 
                          default='twsvd',
                          choices=['twsvd', 'amwsvd'],
                          help='the method to be used')
    myparser.add_argument('-t', '--threshold', 
                          default=0.8,
                          type=float,
                          help='the threshold for the singular values')
    args = myparser.parse_args()

    # run
    main(args.inputfolder, args.nbands, args.method, args.threshold)