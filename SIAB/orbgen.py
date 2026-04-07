# in-built modules
import os
import argparse
import logging

# local modules
from SIAB.io.param import PROGRAM_HEADER, PROGRAM_TAIL
from SIAB.driver.main import init as read
from SIAB.driver.main import rundft, minimize_spillage

def start() -> str:
    '''User interface, return the input json file specified in -i tag'''

    helpmsg =  ''
    helpmsg += 'To generate numerical atomic orbital (NAO) with ABACUS Spherical Wave'
    helpmsg += ' Contraction (ABACUS-CSW-NAO) workflow, you should complie the ABACUS with'
    helpmsg += ' version >= 3.7.5, < 3.9.0.6. Examples can be found in folder examples/.'
    helpmsg += ' For more information on parameter settings, please refer to the'
    helpmsg += ' Github repository:\n'
    helpmsg += '* https://github.com/kirk0830/ABACUS-CSW-NAO\n\n'
    helpmsg += 'This code is not encouraged to be used by typical users. To query the'
    helpmsg += ' orbitals, submit issue to ABACUS Github repository:\n'
    helpmsg += '* https://github.com/deepmodeling/abacus-develop\n'
    
    parser = argparse.ArgumentParser(description=helpmsg)    
    parser.add_argument('-i', '--input', required=True, help='input json file')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 3.0')
    parser.add_argument('-p', '--prefix', default='', help='prefix of the output files')
    parser.add_argument('-o', '--output', help='output directory')
    parser.add_argument('-l', '--log', help='log file')
    args = parser.parse_args()
    
    fout = args.output if args.output else args.prefix
    flog = args.log if args.log else f'{args.prefix}.log'
    
    return {'input': args.input, 'output': fout, 'log': flog}

def main():
    '''main function'''

    ## start the workflow ##
    f, outdir, flog = start().values()
    
    ## start logging ##
    logging.basicConfig(filename=flog, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(PROGRAM_HEADER)
    print(PROGRAM_HEADER, flush=True)
    print(f'ABACUS CSW-NAO employs the logging module, any runtime information/warning/error'
          ' is redirect to the file {flog}. By default it is written in "append" mode.', 
          flush=True)
    
    ## read parameters ##
    glbparam, dftparam, spillparam, compparam, iop = read(f)

    ## run dft calculation ##
    _ = rundft(atomspecies=[{'elem': glbparam['element'],
                             'ecutjy': spillparam.get('ecutjy', dftparam['ecutwfc']),
                             'zval': 0}],
               geoms=spillparam['geoms'], 
               rcuts=glbparam['bessel_nao_rcut'],
               dftparam=dftparam,
               spillguess=spillparam.get('spill_guess'),
               compparam=compparam,
               **iop)
    
    ## run spillage optimization ##
    options = {k: v for k, v in spillparam.items() 
               if k not in 
               ['geoms', 'orbitals', 'primitive_type', 'fit_basis']}
    ## when call spillage, the element is merely a symbol, so we pass it as a string
    minimize_spillage(elem=glbparam['element'],
                      ecut=spillparam['ecutjy'],
                      rcuts=glbparam['bessel_nao_rcut'],
                      primitive_type=spillparam['primitive_type'],
                      scheme=spillparam['orbitals'],
                      dft_root=os.getcwd(),
                      run_mode=spillparam['fit_basis'],
                      outdir=outdir,
                      **iop,
                      **options)
    
    ## terminate the logging ##
    logging.info('') # spacing with the previous log
    logging.info('ABACUS-ORBGEN workflow has been done.')
    logging.info(PROGRAM_TAIL)
    logging.shutdown()
    
    ## print the final message ##
    print('ABACUS-ORBGEN workflow has been done.', flush=True)
    print(PROGRAM_TAIL, flush=True)

if __name__ == '__main__':
    '''entry point if run as a script'''
    main()
