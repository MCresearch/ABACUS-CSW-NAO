'''
functionalities related to ABACUS
'''
# in-built modules
import os
import logging

# local modules
from SIAB.driver.control import OrbgenAssertIn
from SIAB.abacus.io import parse_abacus_dftparam, autoset

##############################################
#           general information              #
##############################################
def VersionCheck(version_1: str, version_2: str) -> bool:
    """compare two version strings, return True if version_1 <= version_2"""
    version_1 = version_1.split(".")
    version_2 = version_2.split(".")
    for i in range(len(version_1)):
        if int(version_1[i]) < int(version_2[i]):
            return True
        elif int(version_1[i]) > int(version_2[i]):
            return False
        else:
            continue
    return True

def DuplicateCheck(folder: str, param: dict):
    """check if the abacus calculation can be safely (really?)
    skipped"""
    logging.info('')
    # STAGE1: existence of folder
    stage = 'DUPLICATE CHECK-1'
    if not os.path.isdir(folder):
        logging.warning(f'{stage} fail: folder {folder} does not exist')
        return False
    logging.info(f'{stage} pass: folder {folder} exists')
    
    # STAGE2: existence of INPUT files
    stage = 'DUPLICATE CHECK-2'
    files = os.listdir(folder)
    if "INPUT" not in files:
        logging.warning(f'{stage} fail: INPUT does not exist')
        return False
    if "INPUTw" not in files:
        # it depends: if it is lcao calculation, INPUTw is not necessary
        if param.get("basis_type", "pw") == "pw":
            logging.warning(f'{stage} fail: INPUTw does not exist')
            return False
    logging.info(f'{stage} pass: INPUT (and INPUTw) exist')
    
    # STAGE3: correspondence of INPUT settings
    stage = 'DUPLICATE CHECK-3'
    OrbgenAssertIn(['bessel_nao_rcut', 'lmaxmax'], param,
        lambda x: f'NECESSARY KEYWORD {x} is not specified')
    param_ = parse_abacus_dftparam(folder)
    param = autoset(param)
    
    check_keys = [k for k in param.keys() 
        if k not in ["orbital_dir", "bessel_nao_rcut"]]
    check_keys = list(param.keys())\
        if param.get("basis_type", "pw") == "pw" else check_keys
    for key in check_keys:
        value = param[key]
        if isinstance(value, list):
            value = " ".join([str(v) for v in value])
        else: # do i need to convert the type to str?
            value = str(value)
        value_ = str(param_.get(key))
        # for jy, it is different here. Because the forb is no where to store, all orbitals
        # involved are temporarily stored in the value of key "orbital_dir". Thus the following
        # will fail for jy for two keys: orbital_dir and bessel_nao_rcut, the latter is because
        # for jy, one SCF can only have one rcut.
        if value_ != value:
            logging.info(f"KEYWORD '{key}' has different values. \
Original: '{value_}', new: '{value}'. Difference detected, start a new job.")
            return False
    
    # for jy, the following will also fail, because jy will not print such matrix, instead, 
    # there will only be several matrices such as T(k), S(k), H(k) and wavefunction file.    
    logging.info("DUPLICATE CHECK-3 pass: INPUT settings are consistent")

    # STAGE4: existence of crucial output files
    rcuts = param["bessel_nao_rcut"]
    rcuts = [rcuts] if not isinstance(rcuts, list) else rcuts
    # logging.info(param_.get("bessel_nao_rcut"))
    if param.get("basis_type", "pw") != "pw" and \
        float(param_.get("bessel_nao_rcut", 0)) in [float(rcut) for rcut in rcuts]:
        logging.info("DUPLICATE CHECK-4 pass: realspace cutoff matches \
(file integrities not checked)")
        return True
    
    if len(rcuts) == 1:
        if "orb_matrix.0.dat" not in files:
            return False
        if "orb_matrix.1.dat" not in files:
            return False
    else:
        for rcut in rcuts:
            if "orb_matrix_rcut%sderiv0.dat"%rcut not in files:
                return False
            if "orb_matrix_rcut%sderiv1.dat"%rcut not in files:
                return False
    logging.info("DUPLICATE CHECK-4 pass: crucial output files exist")
    return True
