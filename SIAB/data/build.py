'''this is just a demo file'''
import unittest
import logging

import numpy as np

from SIAB.driver.control import OrbgenAssert
from SIAB.spillage.radial import _nbes

class AtomSpecies:
    '''Class for atom species'''
    def __init__(self, 
                 elem,
                 mass,
                 fpsp,
                 forb=None,
                 ecutjy=100,
                 rcutjy=10,
                 lmax=3,
                 orbgen=False):
        '''
        instantiate the AtomSpecies object
        
        Parameters
        ----------
        elem : str
            element symbol
        mass : float
            atomic mass, in amu
        fpsp : str
            pseudopotential file path
        forb : str
            the orbital file path. If there is no such file, set it to None
        ecutjy : float
            kinetic energy cutoff for primitive jy basis, in Ry
        rcutjy : float
            cutoff radius for primitive jy basis, in Bohr
        lmax : int
            maximum angular momentum
        orbgen : bool
            whether this element is the one to generate orbital, default is False
        '''
        import os
        OrbgenAssert(os.path.exists(fpsp), 
                     f'Pseudopotential file {fpsp} not found',
                     FileNotFoundError)   
        self.elem = elem
        self.mass = mass
        self.fpsp = fpsp
        self.forb = forb
        self.ecutjy = ecutjy
        self.rcutjy = rcutjy
        self.lmax = lmax
        self.orbgen = orbgen
    
    def jygen(self, outdir, dr=0.01, primitive_type='reduced'):
        '''Generate jy orbital file, return the file path'''
        # in-built modules
        import os
        # local modules
        from SIAB.orb.orb import Orbital
        from SIAB.io.convention import orb as name_orb
        
        nzeta = [_nbes(l, self.rcutjy, self.ecutjy) - int(primitive_type == 'reduce')
                 for l in range(self.lmax+1)]
        fn = name_orb(self.elem, self.rcutjy, self.ecutjy, nzeta)
        self.forb = os.path.join(outdir, fn) if self.forb is None else self.forb
        if self.forb != os.path.join(outdir, fn):
            logging.warning(
                f'overwrite the self.forb ({self.forb}) with {os.path.join(outdir, fn)}')
            self.forb = os.path.join(outdir, fn)
        
        if os.path.exists(os.path.join(outdir, fn)):
            logging.info(f'File {fn} already exists, skip the generation')
            return os.path.join(outdir, fn)
        
        myorb = Orbital(ecut=self.ecutjy, 
                        rcut=self.rcutjy, 
                        elem=self.elem, 
                        nzeta=[0]*(self.lmax+1), # an empty object
                        primitive_type=primitive_type)
        myorb.coef_ = myorb.coefgen('ones') # initialize, not empty now
        _ = myorb.to_griddata(dr=dr, fn=os.path.join(outdir, fn))
        return os.path.join(outdir, fn)
    
    @staticmethod
    def to_index(elem):
        __data = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 
            'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13,
            'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19,
            'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
            'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31,
            'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37,
            'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
            'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49,
            'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55,
            'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61,
            'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67,
            'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73,
            'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79,
            'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
            'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91,
            'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97,
            'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103,
            'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
            'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115,
            'Lv': 116, 'Ts': 117, 'Og': 118
        }
        return __data[elem]

    @staticmethod
    def to_name(elem):
        __data = {
            'H' : 'Hydrogen',      'He': 'Helium',      'Li': 'Lithium' ,       'Be': 'Beryllium', 
            'B' : 'Boron'   ,      'C' : 'Carbon',      'N' : 'Nitrogen',       'O' : 'Oxygen'   ,
            'F' : 'Fluorine',      'Ne': 'Neon'  ,      'Na': 'Sodium',         'Mg': 'Magnesium', 
            'Al': 'Aluminium',     'Si': 'Silicon',     'P' : 'Phosphorus',     'S': 'Sulfur', 
            'Cl': 'Chlorine',      'Ar': 'Argon',       'K' : 'Potassium',      'Ca': 'Calcium', 
            'Sc': 'Scandium',      'Ti': 'Titanium',    'V' : 'Vanadium',       'Cr': 'Chromium', 
            'Mn': 'Manganese',     'Fe': 'Iron',        'Co': 'Cobalt',         'Ni': 'Nickel', 
            'Cu': 'Copper',        'Zn': 'Zinc',        'Ga': 'Gallium',        'Ge': 'Germanium', 
            'As': 'Arsenic',       'Se': 'Selenium',    'Br': 'Bromine',        'Kr': 'Krypton', 
            'Rb': 'Rubidium',      'Sr': 'Strontium',   'Y' : 'Yttrium',        'Zr': 'Zirconium', 
            'Nb': 'Niobium',       'Mo': 'Molybdenum',  'Tc': 'Technetium',     'Ru': 'Ruthenium', 
            'Rh': 'Rhodium',       'Pd': 'Palladium',   'Ag': 'Silver',         'Cd': 'Cadmium', 
            'In': 'Indium',        'Sn': 'Tin',         'Sb': 'Antimony',       'Te': 'Tellurium', 
            'I' : 'Iodine',        'Xe': 'Xenon',       'Cs': 'Caesium',        'Ba': 'Barium', 
            'La': 'Lanthanum',     'Ce': 'Cerium',      'Pr': 'Praseodymium',   'Nd': 'Neodymium', 
            'Pm': 'Promethium',    'Sm': 'Samarium',    'Eu': 'Europium',       'Gd': 'Gadolinium', 
            'Tb': 'Terbium',       'Dy': 'Dysprosium',  'Ho': 'Holmium',
            'Er': 'Erbium',        'Tm': 'Thulium',     'Yb': 'Ytterbium',      'Lu': 'Lutetium', 
            'Hf': 'Hafnium',       'Ta': 'Tantalum',    'W' : 'Tungsten',       'Re': 'Rhenium', 
            'Os': 'Osmium',        'Ir': 'Iridium',     'Pt': 'Platinum',       'Au': 'Gold', 
            'Hg': 'Mercury',       'Tl': 'Thallium',    'Pb': 'Lead',           'Bi': 'Bismuth', 
            'Po': 'Polonium',      'At': 'Astatine',    'Rn': 'Radon',          'Fr': 'Francium', 
            'Ra': 'Radium',        'Ac': 'Actinium',    'Th': 'Thorium',        'Pa': 'Protactinium', 
            'U' : 'Uranium',       'Np': 'Neptunium',   'Pu': 'Plutonium',      'Am': 'Americium', 
            'Cm': 'Curium',        'Bk': 'Berkelium',   'Cf': 'Californium',    'Es': 'Einsteinium', 
            'Fm': 'Fermium',       'Md': 'Mendelevium', 'No': 'Nobelium',       'Lr': 'Lawrencium', 
            'Rf': 'Rutherfordium', 'Db': 'Dubnium',     'Sg': 'Seaborgium',     'Bh': 'Bohrium', 
            'Hs': 'Hassium',       'Mt': 'Meitnerium',  'Ds': 'Darmstadtium',   'Rg': 'Roentgenium', 
            'Cn': 'Copernicium',   'Nh': 'Nihonium',    'Fl': 'Flerovium',      'Mc': 'Moscovium', 
            'Lv': 'Livermorium',   'Ts': 'Tennessine',  'Og': 'Oganesson'
        }
        return __data[elem]

    @staticmethod
    def to_elem(i: int|str):
        __data_by_name = {
            'Hydrogen': 'H', 'Helium': 'He', 'Lithium': 'Li', 'Beryllium': 'Be', 'Boron': 'B', 'Carbon': 'C', 'Nitrogen': 'N',
            'Oxygen': 'O', 'Fluorine': 'F', 'Neon': 'Ne', 'Sodium': 'Na', 'Magnesium': 'Mg', 'Aluminium': 'Al',
            'Silicon': 'Si', 'Phosphorus': 'P', 'Sulfur': 'S', 'Chlorine': 'Cl', 'Argon': 'Ar', 'Potassium': 'K',
            'Calcium': 'Ca', 'Scandium': 'Sc', 'Titanium': 'Ti', 'Vanadium': 'V', 'Chromium': 'Cr', 'Manganese': 'Mn',
            'Iron': 'Fe', 'Cobalt': 'Co', 'Nickel': 'Ni', 'Copper': 'Cu', 'Zinc': 'Zn', 'Gallium': 'Ga',
            'Germanium': 'Ge', 'Arsenic': 'As', 'Selenium': 'Se', 'Bromine': 'Br', 'Krypton': 'Kr', 'Rubidium': 'Rb',
            'Strontium': 'Sr', 'Yttrium': 'Y', 'Zirconium': 'Zr', 'Niobium': 'Nb', 'Molybdenum': 'Mo', 'Technetium': 'Tc',
            'Ruthenium': 'Ru', 'Rhodium': 'Rh', 'Palladium': 'Pd', 'Silver': 'Ag', 'Cadmium': 'Cd', 'Indium': 'In',
            'Tin': 'Sn', 'Antimony': 'Sb', 'Tellurium': 'Te', 'Iodine': 'I', 'Xenon': 'Xe', 'Caesium': 'Cs',
            'Barium': 'Ba', 'Lanthanum': 'La', 'Cerium': 'Ce', 'Praseodymium': 'Pr', 'Neodymium': 'Nd', 'Promethium': 'Pm',
            'Samarium': 'Sm', 'Europium': 'Eu', 'Gadolinium': 'Gd', 'Terbium': 'Tb', 'Dysprosium': 'Dy', 'Holmium': 'Ho',
            'Erbium': 'Er', 'Thulium': 'Tm', 'Ytterbium': 'Yb', 'Lutetium': 'Lu', 'Hafnium': 'Hf', 'Tantalum': 'Ta',
            'Tungsten': 'W', 'Rhenium': 'Re', 'Osmium': 'Os', 'Iridium': 'Ir', 'Platinum': 'Pt', 'Gold': 'Au',
            'Mercury': 'Hg', 'Thallium': 'Tl', 'Lead': 'Pb', 'Bismuth': 'Bi', 'Polonium': 'Po', 'Astatine': 'At',
            'Radon': 'Rn', 'Francium': 'Fr', 'Radium': 'Ra', 'Actinium': 'Ac', 'Thorium': 'Th', 'Protactinium': 'Pa',
            'Uranium': 'U', 'Neptunium': 'Np', 'Plutonium': 'Pu', 'Americium': 'Am', 'Curium': 'Cm', 'Berkelium': 'Bk',
            'Californium': 'Cf', 'Einsteinium': 'Es', 'Fermium': 'Fm', 'Mendelevium': 'Md', 'Nobelium': 'No', 'Lawrencium': 'Lr',
            'Rutherfordium': 'Rf', 'Dubnium': 'Db', 'Seaborgium': 'Sg', 'Bohrium': 'Bh', 'Hassium': 'Hs', 'Meitnerium': 'Mt',
            'Darmstadtium': 'Ds', 'Roentgenium': 'Rg', 'Copernicium': 'Cn', 'Nihonium': 'Nh', 'Flerovium': 'Fl', 'Moscovium': 'Mc',
            'Livermorium': 'Lv', 'Tennessine': 'Ts', 'Oganesson': 'Og'
        }
        __data_by_index = [
            'H',  'He', 
            'Li', 'Be', 'B' , 'C' , 'N', 'O',  'F', 'Ne', 
            'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 'Ar', 
            'K' , 'Ca', 
            'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
            'Rb', 'Sr', 
            'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 
            'In', 'Sn', 'Sb', 'Te', 'I' , 'Xe', 
            'Cs', 'Ba', 
            'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho',
            'Er', 'Tm', 'Yb', 'Lu', 
            'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 
            'Fr', 'Ra', 'Ac', 'Th', 'Pa',
            'U',  'Np', 'Pu', 'Am', 'Cm', 'Bk',
            'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
            'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt',
            'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
            'Lv', 'Ts', 'Og'
        ]
        return __data_by_name[i] if isinstance(i, str) else __data_by_index[i]

    @staticmethod
    def ground_state_atomic_electronic_configuration(elem):
        '''get the ground state electronic configuration of the element.
        The number is the number of electrons in the shell, shells are 
        arranged in order like:
        1s, 
        2s, 2p, 
        3s, 3p, 3d, 
        4s, 4p, 4d, 4f,
        5s, 5p, 5d, 5f, 5g,
        6s, 6p, 6d, 6f, 6g, ...
        
        Parameters
        ----------
        elem : str
            element symbol
        
        Returns
        -------
        list[int]
            the list of electrons in the shell
        '''
        __data = {
            'H' : [1],             'He': [2],             'Li': [2, 1], 
            'Be': [2, 2],          'B' : [2, 2, 1],       'C' : [2, 2, 2], 
            'N' : [2, 2, 3],       'O' : [2, 2, 4],       'F' : [2, 2, 5],
            'Ne': [2, 2, 6],       'Na': [2, 2, 6, 1],    'Mg': [2, 2, 6, 2], 
            'Al': [2, 2, 6, 2, 1], 'Si': [2, 2, 6, 2, 2], 'P': [2, 2, 6, 2, 3], 
            'S' : [2, 2, 6, 2, 4], 'Cl': [2, 2, 6, 2, 5], 'Ar': [2, 2, 6, 2, 6], 
            'K' : [2, 2, 6, 2, 6, 0, 1],     'Ca': [2, 2, 6, 2, 6, 0, 2], 
            'Sc': [2, 2, 6, 2, 6, 1, 2],     'Ti': [2, 2, 6, 2, 6, 2, 2], 
            'V' : [2, 2, 6, 2, 6, 3, 2],     'Cr': [2, 2, 6, 2, 6, 4, 2], 
            'Mn': [2, 2, 6, 2, 6, 5, 2],     'Fe': [2, 2, 6, 2, 6, 6, 2], 
            'Co': [2, 2, 6, 2, 6, 7, 2],     'Ni': [2, 2, 6, 2, 6, 8, 2], 
            'Cu': [2, 2, 6, 2, 6, 10, 1],    'Zn': [2, 2, 6, 2, 6, 10, 2], 
            'Ga': [2, 2, 6, 2, 6, 10, 2, 1], 'Ge': [2, 2, 6, 2, 6, 10, 2, 2], 
            'As': [2, 2, 6, 2, 6, 10, 2, 3], 'Se': [2, 2, 6, 2, 6, 10, 2, 4], 
            'Br': [2, 2, 6, 2, 6, 10, 2, 5], 'Kr': [2, 2, 6, 2, 6, 10, 2, 6], 
            'Rb': [2, 2, 6, 2, 6, 10, 2, 6,  0,  0, 1], 
            'Sr': [2, 2, 6, 2, 6, 10, 2, 6,  0,  0, 2], 
            'Y' : [2, 2, 6, 2, 6, 10, 2, 6,  1,  0, 2], 
            'Zr': [2, 2, 6, 2, 6, 10, 2, 6,  2,  0, 2], 
            'Nb': [2, 2, 6, 2, 6, 10, 2, 6,  4,  0, 1], 
            'Mo': [2, 2, 6, 2, 6, 10, 2, 6,  5,  0, 1], 
            'Tc': [2, 2, 6, 2, 6, 10, 2, 6,  5,  0, 2], 
            'Ru': [2, 2, 6, 2, 6, 10, 2, 6,  7,  0, 1], 
            'Rh': [2, 2, 6, 2, 6, 10, 2, 6,  8,  0, 1], 
            'Pd': [2, 2, 6, 2, 6, 10, 2, 6, 10,  0, 0], 
            'Ag': [2, 2, 6, 2, 6, 10, 2, 6, 10,  0, 1], 
            'Cd': [2, 2, 6, 2, 6, 10, 2, 6, 10,  0, 2], 
            'In': [2, 2, 6, 2, 6, 10, 2, 6, 10,  0, 2, 1], 
            'Sn': [2, 2, 6, 2, 6, 10, 2, 6, 10,  0, 2, 2], 
            'Sb': [2, 2, 6, 2, 6, 10, 2, 6, 10,  0, 2, 3], 
            'Te': [2, 2, 6, 2, 6, 10, 2, 6, 10,  0, 2, 4], 
            'I' : [2, 2, 6, 2, 6, 10, 2, 6, 10,  0, 2, 5], 
            'Xe': [2, 2, 6, 2, 6, 10, 2, 6, 10,  0, 2, 6], 
            'Cs': [2, 2, 6, 2, 6, 10, 2, 6, 10,  0, 2, 6,  0, 0, 0, 1], 
            'Ba': [2, 2, 6, 2, 6, 10, 2, 6, 10,  0, 2, 6,  0, 0, 0, 2], 
            'La': [2, 2, 6, 2, 6, 10, 2, 6, 10,  0, 2, 6,  1, 0, 0, 2], 
            'Ce': [2, 2, 6, 2, 6, 10, 2, 6, 10,  1, 2, 6,  1, 0, 0, 2], 
            'Pr': [2, 2, 6, 2, 6, 10, 2, 6, 10,  3, 2, 6,  0, 0, 0, 2], 
            'Nd': [2, 2, 6, 2, 6, 10, 2, 6, 10,  4, 2, 6,  0, 0, 0, 2], 
            'Pm': [2, 2, 6, 2, 6, 10, 2, 6, 10,  5, 2, 6,  0, 0, 0, 2], 
            'Sm': [2, 2, 6, 2, 6, 10, 2, 6, 10,  6, 2, 6,  0, 0, 0, 2], 
            'Eu': [2, 2, 6, 2, 6, 10, 2, 6, 10,  7, 2, 6,  0, 0, 0, 2], 
            'Gd': [2, 2, 6, 2, 6, 10, 2, 6, 10,  7, 2, 6,  1, 0, 0, 2], 
            'Tb': [2, 2, 6, 2, 6, 10, 2, 6, 10,  9, 2, 6,  0, 0, 0, 2], 
            'Dy': [2, 2, 6, 2, 6, 10, 2, 6, 10, 10, 2, 6,  0, 0, 0, 2], 
            'Ho': [2, 2, 6, 2, 6, 10, 2, 6, 10, 11, 2, 6,  0, 0, 0, 2], 
            'Er': [2, 2, 6, 2, 6, 10, 2, 6, 10, 12, 2, 6,  0, 0, 0, 2], 
            'Tm': [2, 2, 6, 2, 6, 10, 2, 6, 10, 13, 2, 6,  0, 0, 0, 2], 
            'Yb': [2, 2, 6, 2, 6, 10, 2, 6, 10, 14, 2, 6,  0, 0, 0, 2], 
            'Lu': [2, 2, 6, 2, 6, 10, 2, 6, 10, 14, 2, 6,  1, 0, 0, 2], 
            'Hf': [2, 2, 6, 2, 6, 10, 2, 6, 10, 14, 2, 6,  2, 0, 0, 2], 
            'Ta': [2, 2, 6, 2, 6, 10, 2, 6, 10, 14, 2, 6,  3, 0, 0, 2], 
            'W' : [2, 2, 6, 2, 6, 10, 2, 6, 10, 14, 2, 6,  4, 0, 0, 2], 
            'Re': [2, 2, 6, 2, 6, 10, 2, 6, 10, 14, 2, 6,  5, 0, 0, 2], 
            'Os': [2, 2, 6, 2, 6, 10, 2, 6, 10, 14, 2, 6,  6, 0, 0, 2], 
            'Ir': [2, 2, 6, 2, 6, 10, 2, 6, 10, 14, 2, 6,  7, 0, 0, 2], 
            'Pt': [2, 2, 6, 2, 6, 10, 2, 6, 10, 14, 2, 6,  9, 0, 0, 1], 
            'Au': [2, 2, 6, 2, 6, 10, 2, 6, 10, 14, 2, 6, 10, 0, 0, 1], 
            'Hg': [2, 2, 6, 2, 6, 10, 2, 6, 10, 14, 2, 6, 10, 0, 0, 2], 
            'Tl': [2, 2, 6, 2, 6, 10, 2, 6, 10, 14, 2, 6, 10, 0, 0, 2, 1], 
            'Pb': [2, 2, 6, 2, 6, 10, 2, 6, 10, 14, 2, 6, 10, 0, 0, 2, 2], 
            'Bi': [2, 2, 6, 2, 6, 10, 2, 6, 10, 14, 2, 6, 10, 0, 0, 2, 3], 
            'Po': [2, 2, 6, 2, 6, 10, 2, 6, 10, 14, 2, 6, 10, 0, 0, 2, 4], 
            'At': [2, 2, 6, 2, 6, 10, 2, 6, 10, 14, 2, 6, 10, 0, 0, 2, 5], 
            'Rn': [2, 2, 6, 2, 6, 10, 2, 6, 10, 14, 2, 6, 10, 0, 0, 2, 6]
        }
        return __data[elem]

    @staticmethod
    def cal_slater_screening_coef(elem, n, l):
        '''calculate the slater screening constant for the configuration
        
        Parameters
        ----------
        elem : str
            element symbol
        n : int
            the principal quantum number
        l : int
            the angular momentum quantum number
        
        Notes
        -----
        The order of conf should be consistent with the one in function
        `ground_state_atomic_electronic_configuration`!
        
        Returns
        -------
        float
            the slater screening constant
        '''
        # short cut for n = 1 and H, He
        if elem == 'H':
            return 0.00
        elif elem == 'He':
            return 0.30
        elif n == 1:
            return 0.30
        # otherwise...
        conf = AtomSpecies.ground_state_atomic_electronic_configuration(elem)
        sigma = 0.0
        isubshell = 0
        for n_ in range(1, n + 1): # the real principle quantum number
            lmax = l if n_ == n else n_ - 1
            for l_ in range(lmax + 1):
                if isubshell >= len(conf):
                    break # all the electrons are considered, we can stop
                nelec = conf[isubshell]
                if n - n_ >= 2:
                    sigma += nelec * 1.0
                elif n - n_ == 1:
                    sigma += nelec * (1.00 if l > 1 else 0.85)
                else:
                    if l_ == l:
                        sigma += (nelec - 1) * 0.35
                    else:
                        sigma += nelec * (1.00 if l > 1 else 0.35)
                isubshell += 1
        return sigma

    @staticmethod
    def build_hydrogen_orb(elem, n, l, r, slater=True):
        '''build the radial function of hydrogen-like orbital
        
        Parameters
        ----------
        elem : str
            element symbol
        n : int
            the principal quantum number
        l : int
            the angular momentum quantum number
        r : np.ndarray
            the radial coordinate in Bohr
        slater : bool
            whether to use slater screening constant
        
        Returns
        -------
        np.ndarray
            the radial function
            
        Notes
        -----
        this function's implementation is from ABACUS hydrogen_radials.cpp
        https://github.com/deepmodeling/abacus-develop/blob/develop/source/module_basis/module_nao/hydrogen_radials.cpp, whose author is also myself.
        '''
        from math import factorial
        from scipy.special import genlaguerre
        a0 = 1 # in bohr
        
        z = AtomSpecies.to_index(elem)
        sigma = AtomSpecies.cal_slater_screening_coef(elem, n, l)\
            if slater else 0.0
        z -= sigma
        rho = 2 * z * r / n / a0
        f = np.sqrt((2 * z / n)**3 
                    * factorial(n - l - 1) 
                    / (2 * n * factorial(n + l)) 
                    / a0**3)
        laguerre = genlaguerre(n - l - 1, 2 * l + 1)(rho)
        return f * np.exp(-rho / 2) * (rho**l) * laguerre

    @staticmethod
    def get_covalent_radius(elem):
        __data = {
            'Fr': 2.6, 'Cs': 2.44, 'Ra': 2.21, 'Rb': 2.2, 'Ac': 2.15, 'Ba': 2.15, 'La': 2.07, 
            'Th': 2.06, 'Ce': 2.04, 'Pr': 2.03, 'K': 2.03, 
            'Nd': 2.01, 'Pa': 2.0, 'Pm': 1.99, 'Eu': 1.98, 'Sm': 1.98, 'U': 1.96, 
            'Gd': 1.96, 'Sr': 1.95, 'Tb': 1.94, 'Ho': 1.92, 'Dy': 1.92, 'Np': 1.90, 
            'Tm': 1.90, 'Y': 1.90, 'Er': 1.89, 
            'Pu': 1.87, 'Lu': 1.87, 'Yb': 1.87, 'Am': 1.8, 'Ca': 1.76, 'Hf': 1.75, 'Zr': 1.75, 
            'Ta': 1.7, 'Sc': 1.7, 'Cm': 1.69, 'Na': 1.66, 'Nb': 1.64, 
            'W': 1.62, 'Ti': 1.6, 'Mo': 1.54, 'V': 1.53, 'Re': 1.51, 'Rn': 1.5, 'At': 1.5, 
            'Bi': 1.48, 'Tc': 1.47, 'Pb': 1.46, 'Ru': 1.46, 'Tl': 1.45, 'Ag': 1.45, 'Os': 1.44, 
            'Cd': 1.44, 'In': 1.42, 'Rh': 1.42, 'Ir': 1.41, 'Mg': 1.41, 'Po': 1.40, 
            'Xe': 1.40, 'I': 1.39, 'Sb': 1.39, 
            'Sn': 1.39, 'Pd': 1.39, 'Mn': 1.39, 
            'Cr': 1.39, 'Te': 1.38, 'Au': 1.36, 'Pt': 1.36, 'Hg': 1.32, 
            'Cu': 1.32, 'Fe': 1.32, 'Li': 1.28, 'Co': 1.26, 'Ni': 1.24, 'Ga': 1.22, 'Zn': 1.22, 
            'Al': 1.21, 'Br': 1.2, 'Se': 1.2, 'Ge': 1.2, 'As': 1.19, 'Kr': 1.16, 'Si': 1.11, 'P': 1.07, 
            'Ar': 1.06, 'S': 1.05, 'Cl': 1.02, 'Be': 0.96, 'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66, 
            'Ne': 0.58, 'F': 0.57, 'H': 0.31, 'He': 0.28
        }
        return __data.get(elem, None)
    
    def _zero_charge_van_der_waals_radius(elem):
        '''
        Komissarov, A.V. and Heaven, M.C., J. Chem. Phys., 2000, vol. 113, no. 5, pp. 1775-1780
        '''
        __data = {
            'H': 1.96, 'C': 1.85, 'Li': 2.72, 'Si': 2.25, 'Na': 2.82, 'Ge': 2.23, 
            'K': 3.08, 'Sn': 2.34, 'Rb': 3.22, 'Pb': 2.34, 'Cs': 3.38, 'Nb': 2.5, 
            'Cu': 2.3, 'Ta': 2.44, 'Ag': 2.34, 'N': 1.7, 'Be': 2.32, 'P': 2.09, 
            'Mg': 2.45, 'As': 2.16, 'Ca': 2.77, 'Sb': 2.33, 'Sr': 2.9, 'Bi': 2.4, 
            'Ba': 3.05, 'Cr': 2.23, 'Zn': 2.25, 'Mo': 2.4, 'Cd': 2.32, 'W': 2.35, 
            'Hg': 2.25, 'O': 1.64, 'Sc': 2.64, 'S': 2.0, 'Y': 2.73, 'Se': 2.1, 
            'La': 2.86, 'Te': 2.3, 'B': 2.05, 'Mn': 2.29, 'Al': 2.47, 'Re': 2.38, 
            'Ga': 2.38, 'Br': 2.0, 'In': 2.44, 'I': 2.15, 'Tl': 2.46, 'Fe': 2.34, 
            'Ti': 2.52, 'Co': 2.3, 'Zr': 2.63, 'Ni': 2.26, 'Hf': 2.54, 'Th': 2.78, 'U': 2.8
        }
        return __data.get(elem, None)

    def _equlibrium_van_der_waals_radius(elem):
        '''
        Komissarov, A.V. and Heaven, M.C., J. Chem. Phys., 2000, vol. 113, no. 5, pp. 1775-1780
        '''
        __data = {
            'H': 1.56, 'C': 1.97, 'Li': 2.46, 'Si': 2.27, 'Na': 2.68, 'Ge': 2.42, 'K': 3.07, 
            'Sn': 2.57, 'Rb': 3.23, 'Pb': 2.72, 'Cs': 3.42, 'Nb': 2.41, 'Cu': 2.24, 'Ta': 2.41, 
            'Ag': 2.41, 'N': 1.88,  'Be': 2.14, 'P': 2.2, 'Mg': 2.41, 'As': 2.34, 'Ca': 2.79, 
            'Sb': 2.5, 'Sr': 2.98, 'Bi': 2.64, 'Ba': 3.05, 'Cr': 2.23, 'Zn': 2.27, 'Mo': 2.37, 
            'Cd': 2.48, 'W': 2.37, 'Hg': 2.51, 'O': 1.78, 'Sc': 2.59, 'S': 2.13, 'Y': 2.69, 
            'Se': 2.27, 'La': 2.76, 'Te': 2.42, 'B': 2.06, 'Mn': 2.22, 'Al': 2.34, 'Re': 2.35, 
            'Ga': 2.44, 'Br': 2.2, 'In': 2.62, 'I': 2.34, 'Tl': 2.57, 'Fe': 2.21, 
            'Ti': 2.37, 'Co': 2.21, 'Zr': 2.52, 'Ni': 2.2, 'Hf': 2.51, 'Th': 2.72, 'U': 2.5
        }
        return __data.get(elem, None)

    def _crystal_van_der_waals_radius(elem):
        __data = {
            'H': 1.2, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98, 'O': 1.52, 'S': 1.8, 
            'N': 1.55, 'C': 1.7,  'Li': 2.24, 'Na': 2.57, 'K': 3.0, 'Rb': 3.12, 'Cs': 3.31, 
            'Cu': 2.0, 'Ag': 2.13, 'Au': 2.13, 'Be': 1.86, 'Mg': 2.27, 'Ca': 2.61, 'Sr': 2.78, 
            'Ba': 2.85, 'Zn': 2.02, 'Cd': 2.17, 'Hg': 2.17, 'Sc': 2.28, 'Y': 2.45, 'La': 2.51, 
            'B': 1.74, 'Al': 2.11, 'Ga': 2.08, 'In': 2.24, 'Tl': 2.25, 'Ti': 2.14, 'Zr': 2.25, 
            'Hf': 2.24, 'Si': 2.06, 'Ge': 2.13, 'Sn': 2.29, 'Pb': 2.36, 'V': 2.03, 'Nb': 2.13, 
            'Ta': 2.13, 'As': 2.16, 'Sb': 2.33, 'Bi': 2.42, 'Cr': 1.97, 'Mo': 2.06, 'W': 2.07, 
            'Mn': 1.96, 'Tc': 2.04, 'Re': 2.05, 'Fe': 1.96, 'Co': 1.95, 'Ni': 1.94, 'Ru': 2.02, 
            'Rh': 2.02, 'Pd': 2.05, 'Os': 2.03, 'Ir': 2.03, 'Pt': 2.06, 'Th': 2.43, 'U': 2.17
        }
        return __data.get(elem, None)

    def _theoretical_van_der_waals_radius(elem):
        __data = {
            'Li': 1.9, 'B': 1.2, 'P': 1.63, 'Br': 1.73, 'Na': 2.32, 'Al': 1.75, 
            'As': 1.81, 'I': 1.98, 'K': 2.88, 'Ga': 1.75, 'Sb': 2.03, 'Mn': 1.66, 
            'Rb': 3.04, 'In': 1.96, 'Bi': 2.17, 'Tc': 1.73, 'Cs': 3.27, 'Tl': 1.98, 
            'V': 1.72, 'Re': 1.75, 'Cu': 1.73, 'Sc': 1.98, 'Nb': 1.86, 'Fe': 1.65, 
            'Ag': 1.77, 'Y': 2.2, 'Ta': 1.87, 'Co': 1.64, 'Au': 1.86, 'La': 2.29, 
            'S': 1.73, 'Ni': 1.63, 'Be': 1.38, 'Si': 1.68, 'Se': 1.9, 'Ru': 1.81, 
            'Mg': 1.96, 'Ge': 1.77, 'Te': 2.14, 'Rh': 1.75, 'Ca': 2.41, 'Sn': 1.99, 
            'Cr': 1.67, 'Pd': 1.86, 'Sr': 2.63, 'Pb': 2.09, 'Mo': 1.76, 'Os': 1.83, 
            'Ba': 2.71, 'Ti': 1.8, 'W': 1.77, 'Ir': 1.77, 'Zn': 1.77, 'Zr': 1.96, 
            'Th': 2.2, 'Pt': 1.87, 'Cd': 1.98, 'Hf': 1.94, 'U': 2.14, 'Hg': 1.98
        }
        return __data.get(elem, None)
    
    def _searched_van_der_waals_radius(elem):
        __data = {
            'He': 1.4, 'Ne': 1.54, 'Ar': 1.88, 'Kr': 2.02, 'Xe': 2.16, 'Rn': 2.3, 'H': 1.2,
            'Li': 1.82, 'Na': 2.27, 'K': 2.75, 'Rb': 3.03, 'Cs': 3.43, 'Fr': 3.48, 'Be': 1.53,
            'Mg': 1.73, 'Ca': 2.31, 'Sr': 2.49, 'Ba': 2.68, 'Ra': 2.83, 'B': 1.92, 'Al': 1.84,
            'Ga': 1.87, 'In': 1.93, 'Tl': 1.96, 'C': 1.7, 'Si': 2.1, 'Ge': 2.11, 'Sn': 2.17,
            'Pb': 2.2, 'N': 1.55, 'P': 1.8, 'As': 1.85, 'Sb': 2.06, 'Bi': 2.07, 'O': 1.52,
            'S': 1.8, 'Se': 1.9, 'Te': 2.06, 'Po': 1.97, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85,
            'I': 1.98, 'At': 2.02, 'U': 1.86
        }
        return __data.get(elem, None)

    def _interp_van_der_waals_radius(elem, field = 'zero_charge'):
        '''for the case that the element is not in the database,
        we do linear interpolation between the two closest elements
        
        Parameters
        ----------
        elem : str
            the element symbol
        field : str
            the type of van der waals radius, one of the following:
            zero_charge, equlibrium, crystal, theoretical, searched
            . default is zero_charge
        
        Returns
        -------
        radius : float
            the van der waals radius of the element
        '''
        call = {'zero_charge': AtomSpecies._zero_charge_van_der_waals_radius,
                'equlibrium': AtomSpecies._equlibrium_van_der_waals_radius,
                'crystal': AtomSpecies._crystal_van_der_waals_radius,
                'theoretical': AtomSpecies._theoretical_van_der_waals_radius,
                'searched': AtomSpecies._searched_van_der_waals_radius}
        
        OrbgenAssert(field in call, f'Field {field} not supported')
        
        out = call[field](elem)
        if out is not None:
            return out

        elem_ = None
        i = AtomSpecies.to_index(elem)
        while elem_ is None and i < 118:
            elem_ = AtomSpecies.to_elem(i)
            i += 1
        x2 = i
        y2 = call[field](AtomSpecies.to_elem(x2))
        elem_ = None
        i = AtomSpecies.to_index(elem)
        while elem_ is None and i > 0:
            elem_ = AtomSpecies.to_elem(i)
            i -= 1
        x1 = i
        y1 = call[field](AtomSpecies.to_elem(x1))
        x = AtomSpecies.to_index(elem)
        return (x - x1) / (x2 - x1) * (y2 - y1) + y1
    
    @staticmethod
    def get_van_der_waals_radius(elem, field = 'zero_charge'):
        ''''''
        return AtomSpecies._interp_van_der_waals_radius(elem, field)

    @staticmethod
    def aufbau(n):
        '''
        estimate the ground state atomic electron configuration according
        to the Aufbau principle
        '''
        def _mul(s):
            spectrum = 'spdfghiklmnoqrtuvwxyz'
            return 2*spectrum.index(s)+1
        
        __order = ['1s', 
                   '2s', '2p', 
                   '3s', '3p', 
                   '4s', '3d', '4p', 
                   '5s', '4d', '5p', 
                   '6s', '4f', '5d', '6p', 
                   '7s', '5f', '6d', '7p']
        m = n
        out = {}
        while m > 0:
            for layer in __order:
                out[layer] = min(m, _mul(layer[-1])*2)
                m -= _mul(layer[-1])*2
                if m <= 0:
                    break
        return out

    @staticmethod
    def cal_nelec_that_can_fill(elem, fill, zval=0):
        '''
        calculate the number of electrons needed that can fill
        the given shell configuration. This function is particularly
        useful when to determine/autoset the number of nbands to
        calculate in DFT calculation:
        >>> import numpy as np
        >>> from SIAB.io.psp import PspParser
        >>> # instantiate an AtomSpecies instance
        >>> a = AtomSpecies(..., fpsp='Si_ONCV_PBE-1.0.upf')
        >>> # instantiate a PspParser instance
        >>> p = PspParser(a.fpsp)
        >>> # calculate the maximal number of zeta functions needed to
        >>> # generate in the whole orbital generation task
        >>> nzmax = np.max(np.array([o['nzeta'] for o in orbitals]))
        >>> # calculate the number of electrons needed to fill the given
        >>> # number of shells
        >>> n = AtomSpecies.cal_nelec_that_can_fill(a.elem, nzmax, p.zval)
        
        Parameters
        ----------
        elem : str
            the element symbol
        fill : list[int]
            the number of additional shells to be filled of each
            angular momentum
        zval : int
            the number of valence electrons in pseudopotential,
            optional, default is 0
        
        Returns
        -------
        n : int
            the number of electrons needed
        '''
        __order = ['1s', 
                   '2s', 
                   '2p', '3s', 
                   '3p', '4s', 
                   '3d', '4p', '5s', 
                   '4d', '5p', '6s', 
                   '4f', '5d', '6p', '7s', 
                   '5f', '6d', '7p', '8s',
                   '5g', '6f', '7d', '8p', '9s']
        __spectrum = 'spdfghiklmnoqrtuvwxyz'
        def _find_outermost(s):
            '''from symbol list, find the shell outermost'''
            return __order[max([__order.index(x) for x in s])]
        def _fill_up_to(s):
            '''return the number of electrons that can fill up to the shell'''
            def _mul(s):
                return (__spectrum.index(s) * 2 + 1) * 2
            OrbgenAssert(s in __order, f'Shell {s} not recognized')
            n = 0
            for o in __order:
                n += _mul(o[-1])
                if o == s:
                    break
            return n
        def _get_shells(n, from_='1s', exclude_=True):
            spectrum = 'spdfghiklmnoqrtuvwxyz'
            out = []
            for o in __order[__order.index(from_)+int(exclude_):]:
                l = spectrum.index(o[-1])
                if l < len(n) and n[l] > 0:
                    out.append(o)
                    n[l] -= 1
            return out

        z = AtomSpecies.to_index(elem)
        coreconf = AtomSpecies.aufbau(z-zval)
        return _fill_up_to(_find_outermost(_get_shells(
            n=fill, from_=list(coreconf.keys())[-1]))) - z + zval

class Cell:
    '''Class for cell'''
    def __init__(self,
                 atomspecies,
                 type_map,
                 cell,
                 coords):
        import numpy as np
        OrbgenAssert(np.array(cell).shape == (3, 3),
                     'Cell should be a 3x3 matrix')
        self.atomspecies = atomspecies
        self.type_map = type_map
        self.cell = np.array(cell)
        self.coords = np.array(coords)

    
    def write(self, fn, fmt='abacus'):
        '''
        '''
        OrbgenAssert(fmt in ['abacus'], 
                     f'Format {fmt} not supported',
                     NotImplementedError)
        with open(fn, 'w') as f:
            f.write(self._write_core_abacus())

        return self
    
    def run(self, dftparam):
        '''
        
        '''
    
    def _write_core_abacus(self, orbital_dir):
        '''
        write the information of cell to abacus input file
        
        Returns
        -------
        out : str
            the content of the abacus input file
        '''
        import numpy as np
        out = ''
        out += 'ATOMIC_SPECIES\n'

        assert all([atom.forb is not None or atom.orbgen for atom in self.atomspecies]), \
            'illegal atom species configuration for `forb` and `orbgen`'
        for a in self.atomspecies:
            out += f'{a.elem} {a.mass} {a.fpsp}\n'
        out += '\n'
        if any([atom.forb for atom in self.atomspecies]):
            out += 'NUMERICAL_ORBITAL\n'
            for a in self.atomspecies:
                if a.forb is None:
                    a.forb = a.jygen(orbital_dir)
                out += f'{a.forb}\n'
            out += '\n'
        out += 'LATTICE_CONSTANT\n1.8897259886\n' # 1 Angstrom in a.u.
        out += 'LATTICE_VECTORS\n'
        for i in range(3):
            out += ' '.join([str(x) for x in self.cell[i]]) + '\n'
        out += 'ATOMIC_POSITIONS\nCartesian_angstrom_center_xyz\n'
        
        idx = np.argsort(self.type_map)

class CellGenerator:
    '''Class for generating Cell'''
    def __init__(self, 
                 proto,
                 atomspecies,
                 types,
                 type_map,
                 dftparam,
                 compparam):
        self.proto = proto
        self.atomspecies = atomspecies
        self.types = types
        self.type_map = type_map
        self.dftparam = dftparam
        self.compparam = compparam
        # hidden variables
        self.cells = []
    
    def run(self):
        '''run dft calculation, return the jobdir'''
        out = []
        
        return out
    
    def perturb(self, pertkind, pertmags):
        '''configure a perturbation task'''
        self.cells.append([self._stretch(self.proto, pertkind, m) for m in pertmags])
        return self
    
    def _stretch(self):
        
        return self

    
    @staticmethod
    def z2cart(zmat):
        '''convert the z-matrix to cartesian coordinates
        
        the zmat should be organized in the following way:
        zmat = [{},
                {'i': 0, 'bl': 1.2},
                {'i': 1, 'bl': 1.2, 'j': 0, 'angl': 90.0},
                {'i': 2, 'bl': 1.2, 'j': 1, 'angl': 90.0, 'k': 0, 'dihl': 180.0},
                {'i': 3, 'bl': 1.2, 'j': 1, 'angl': 90.0, 'k': 2, 'dihl': 180.0},
                ...]
        
        Parameters
        ----------
        zmat : list
            the z-matrix of the molecule, see the above example
        
        Returns
        -------
        coords : np.ndarray
            the cartesian coordinates of the molecule, in shape of (natom, 3)
        '''
        import numpy as np
        coords = []
        for idx, atom in enumerate(zmat):
            if idx == 0:
                coords.append([0.0, 0.0, 0.0])
            elif idx == 1:
                bl = atom['bl']
                coords.append([bl, 0.0, 0.0])
            elif idx == 2:
                i = atom['i']
                bl = atom['bl']
                angl = np.deg2rad(atom['angl'])
                x = bl * np.cos(angl)
                y = bl * np.sin(angl)
                coords.append([x, y, 0.0])
            else:
                i = atom['i']
                j = atom['j']
                k = atom['k']
                bl = atom['bl']
                angl = np.deg2rad(atom['angl'])
                dihl = np.deg2rad(atom['dihl'])

                v1 = np.array(coords[i])
                v2 = np.array(coords[j])
                v3 = np.array(coords[k])

                v1v2 = v1 - v2
                v2v3 = v2 - v3

                n1 = np.cross(v1v2, v2v3)
                n1 /= np.linalg.norm(n1)

                n2 = np.cross(n1, v2v3)
                n2 /= np.linalg.norm(n2)

                d = bl * np.cos(angl)
                h = bl * np.sin(angl)

                v4 = v3 + d * (v2v3 / np.linalg.norm(v2v3)) + h * (np.cos(dihl) * n1 + np.sin(dihl) * n2)
                coords.append(v4.tolist())

        return np.array(coords)
    
    @staticmethod
    def from_simple_molecule(atom, lat, shape, bl):
        '''
        build an instance of Cell from a molecule
        
        Parameters
        ----------
        atom : AtomSpecies
            the atom species, see the above AtomSpecies class
        lat : float
            the lattice constant, in Angstrom
        shape : str
            the shape of the molecule, one of the following:
            monomer, dimer, trimer, tetrahedron, square, triangular_bipyramid, octahedron, cube
        bl : float
            the bond length, in Angstrom
            
        Returns
        -------
        cell : Cell
            the instance of Cell
        '''
        import numpy as np
        def dimer(bl):
            return np.array([[0.0, 0.0, 0.0], 
                             [0.0, 0.0, bl]])
        def trimer(bl):
            return np.array([[0.0, 0.0, 0.0], 
                             [0.0, 0.0, bl], 
                             [0.0, bl*0.86603, bl*0.5]])
        def tetrahedron(bl):
            return np.array([[0.0, 0.0, 0.0], 
                             [0.0, 0.0, bl], 
                             [0.0, bl*0.86603, bl*0.5], 
                             [bl*0.81649, bl*0.28867, bl*0.5]])
        def square(bl):
            return np.array([[0.0, 0.0, 0.0], 
                             [0.0, 0.0, bl], 
                             [bl, 0.0, 0.0], 
                             [bl, 0.0, bl]])
        def triangular_bipyramid(bl):
            return np.array([[bl/1.73205, 0.0, 0.0], 
                             [-bl/1.73205/2, bl/2, 0.0], 
                             [-bl/1.73205/2, -bl/2, 0.0], 
                             [0.0, 0.0, bl*(2/3)**(1/2)], 
                             [0.0, 0.0, -bl*(2/3)**(1/2)]])
        def octahedron(bl):
            return np.array([[bl/2, bl/2, 0.0], 
                             [-bl/2, -bl/2, 0.0], 
                             [bl/2, -bl/2, 0.0], 
                             [-bl/2, bl/2, 0.0], 
                             [0.0, 0.0, bl/2**(1/2)], 
                             [0.0, 0.0, -bl/2**(1/2)]])
        def cube(bl):
            return np.array([[bl/2, bl/2, bl/2], 
                             [-bl/2, -bl/2, bl/2], 
                             [bl/2, -bl/2, bl/2], 
                             [-bl/2, bl/2, bl/2], 
                             [bl/2, bl/2, -bl/2], 
                             [-bl/2, -bl/2, -bl/2], 
                             [bl/2, -bl/2, -bl/2], 
                             [-bl/2, bl/2, -bl/2]])
        
        builder = {'monomer': dimer, 'dimer': dimer, 'trimer': trimer, 
                   'tetrahedron': tetrahedron, 'square': square, 
                   'triangular_bipyramid': triangular_bipyramid, 
                   'octahedron': octahedron, 'cube': cube}        
        coords=builder[shape](bl)
        return Cell(atomspecies=[atom], 
                    type_map=[0]*len(coords), 
                    cell=np.eye(3) * lat, 
                    coords=coords)
        
    @staticmethod
    def get_dimer_bond_length(elem):
        __data = {
            'H': [0.6, 0.75, 0.9, 1.2, 1.5], 'He': [1.25, 1.75, 2.4, 3.25], 
            'Li': [1.5, 2.1, 2.5, 2.8, 3.2, 3.5, 4.2], 'Be': [1.75, 2.0, 2.375, 3.0, 4.0], 
            'B': [1.25, 1.625, 2.5, 3.5], 
            'C': [1.0, 1.25, 1.5, 2.0, 3.0], 'N': [1.0, 1.1, 1.5, 2.0, 3.0],
            'O': [1.0, 1.208, 1.5, 2.0, 3.0], 
            'F': [1.2, 1.418, 1.75, 2.25, 3.25], 'Fm': [1.98, 2.375, 2.75, 3.25, 4.25], 
            'Md': [2.08, 2.5, 3.0, 3.43, 4.25], 
            'No': [2.6, 3.125, 3.75, 4.27, 5.0], 'Ne': [1.5, 1.75, 2.25, 2.625, 3.0, 3.5], 
            'Na': [2.05, 2.4, 2.8, 3.1, 3.3, 3.8, 4.3], 
            'Mg': [2.125, 2.375, 2.875, 3.375, 4.5], 'Al': [2.0, 2.5, 3.0, 3.75, 4.5], 
            'Si': [1.75, 2.0, 2.25, 2.75, 3.75], 
            'P': [1.625, 1.875, 2.5, 3.25, 4.0], 'S': [1.6, 1.9, 2.5, 3.25, 4.0], 
            'Cl': [1.65, 2.0, 2.5, 3.25, 4.0], 
            'Ar': [2.25, 2.625, 3.0, 3.375, 4.0], 'K': [1.8, 2.6, 3.4, 3.8, 4.0, 4.4, 4.8], 
            'Ca': [2.5, 3.0, 3.5, 4.0, 5.0], 
            'Sc': [1.75, 2.15, 2.75, 3.5, 4.5], 'Ti': [1.6, 1.85, 2.5, 3.25, 4.25], 
            'V': [1.45, 1.65, 2.25, 3.0, 4.0], 
            'Cr': [1.375, 1.55, 2.0, 2.75, 3.75], 'Mn': [1.4, 1.6, 2.1, 2.75, 3.75], 
            'Fe': [1.45, 1.725, 2.25, 3.0, 4.0], 
            'Co': [1.8, 2.0, 2.5, 3.5], 'Ni': [1.65, 2.0, 2.5, 3.0, 4.0], 
            'Cu': [1.8, 2.2, 3.0, 4.0], 
            'Zn': [2.0, 2.3, 2.85, 3.5, 4.25], 'Ga': [1.85, 2.1, 2.45, 3.0, 4.0], 
            'Ge': [1.8, 2.0, 2.35, 3.0, 4.0], 
            'As': [1.75, 2.1, 2.5, 3.0, 4.0], 'Se': [1.85, 2.15, 2.5, 3.0, 4.0], 
            'Br': [1.9, 2.25, 2.75, 3.25, 4.0], 
            'Kr': [2.4, 3.0, 3.675, 4.25, 5.0], 'Rb': [2.45, 3.0, 4.0, 5.0], 
            'Sr': [2.75, 3.5, 4.4, 5.0], 
            'Y': [2.125, 2.5, 2.875, 3.25, 4.0, 5.0], 'Zr': [1.9, 2.25, 3.0, 4.0], 
            'Nb': [1.75, 2.05, 2.4, 3.0, 4.0], 
            'Mo': [1.675, 1.9, 2.375, 3.0, 4.0], 'Tc': [1.7, 1.915, 2.375, 3.0, 4.0], 
            'Ru': [1.725, 1.925, 2.375, 3.0, 4.0], 
            'Rh': [1.8, 2.1, 2.5, 3.0, 4.0], 'Pd': [2.0, 2.275, 2.75, 3.75], 
            'Ag': [2.1, 2.45, 3.0, 4.0], 
            'Cd': [2.15, 2.5, 3.1, 4.0, 5.0], 'In': [2.15, 2.5, 3.0, 3.75, 4.75], 
            'Sn': [2.1, 2.4, 3.75, 3.5, 4.5], 
            'Sb': [2.1, 2.5, 3.0, 3.5, 4.5], 'Te': [2.15, 2.55, 3.1, 3.6, 4.5], 
            'I': [2.22, 2.65, 3.25, 4.25], 
            'Xe': [3.0, 3.5, 4.06, 4.5, 5.25], 'Cs': [2.7, 3.5, 4.5, 5.5], 
            'Ba': [2.65, 3.0, 3.5, 4.4, 5.5], 
            'La': [2.2, 2.6, 3.25, 4.0, 5.0], 'Ce': [2.0, 2.375, 2.875, 3.5, 4.5], 
            'Pr': [1.9, 2.25, 2.75, 3.5, 4.5], 
            'Nd': [1.8, 2.125, 2.625, 3.375, 4.5], 'Pm': [1.775, 2.05, 2.5, 3.25, 4.25], 
            'Sm': [1.775, 2.05, 2.5, 3.25, 4.25], 
            'Eu': [1.775, 2.075, 2.5, 3.25, 4.25], 'Gd': [1.8, 2.11, 2.625, 3.375, 4.1, 5.0], 
            'Tb': [1.825, 2.16, 2.625, 3.375, 4.1, 5.0], 
            'Dy': [1.85, 2.24, 2.625, 3.375, 4.1, 5.0], 'Ho': [1.93, 2.375, 3.0, 4.1, 5.0], 
            'Er': [2.025, 2.5, 3.125, 4.1, 5.0], 
            'Tm': [2.2, 2.625, 3.25, 4.1, 5.0], 'Yb': [2.5, 3.0, 3.5, 4.1, 5.0], 
            'Lu': [2.2, 2.5, 3.04, 4.0, 5.0], 
            'Hf': [1.975, 2.49, 3.25, 4.5], 'Ta': [1.85, 2.12, 2.625, 3.25, 4.5], 
            'W': [1.775, 1.99, 2.5, 3.25, 4.5], 
            'Re': [1.775, 2.01, 2.5, 3.25, 4.25], 'Os': [1.8, 2.04, 2.5, 3.25, 4.5], 
            'Ir': [1.85, 2.125, 2.5, 3.25, 4.25], 
            'Pt': [2.0, 2.275, 2.75, 3.75], 'Au': [2.1, 2.45, 3.0, 4.0], 
            'Hg': [2.225, 2.5, 3.04, 4.0, 5.0], 
            'Tl': [2.21, 2.6, 3.11, 3.75, 4.75], 'Pb': [2.225, 2.5, 2.88, 3.625, 4.5], 
            'Bi': [2.225, 2.61, 3.125, 3.75, 4.75], 
            'Po': [2.3, 2.72, 3.25, 3.875, 4.75], 'At': [2.375, 2.83, 3.5, 4.5], 
            'Rn': [2.8, 3.5, 4.17, 4.75, 5.5], 
            'Fr': [2.85, 3.5, 4.43, 5.5], 'Ra': [3.15, 3.5, 4.25, 5.12, 6.0], 
            'Ac': [2.48, 3.1, 3.72, 4.25, 5.0], 
            'Th': [2.25, 2.65, 3.25, 4.0, 5.0], 'Pa': [2.04, 2.3, 3.0, 3.75, 4.75], 
            'U': [1.89, 2.09, 2.75, 3.5, 4.5], 
            'Np': [1.84, 2.05, 2.625, 3.375, 4.5], 'Pu': [1.81, 2.02, 2.5, 3.25, 4.25], 
            'Am': [1.81, 2.03, 2.5, 3.25, 4.25], 
            'Cm': [1.83, 2.07, 2.5, 3.25, 4.25], 'Bk': [1.86, 2.12, 2.5, 3.0, 4.0], 
            'Cf': [1.89, 2.19, 2.625, 3.125, 4.0], 
            'Es': [1.93, 2.29, 2.625, 3.125, 4.0]
        }
        return __data.get(elem, None)
    
    @staticmethod
    def get_trimer_bond_length(elem):
        __data = {
            'S': [1.7, 2.2, 2.8], 'Pd': [2.2, 2.6, 3.2], 'Si': [1.9, 2.1, 2.6], 
            'Te': [2.4, 2.8, 3.4], 'Sn': [2.3, 2.6, 3.1], 'Xe': [3.8, 4.3, 5.0], 
            'Mo': [1.8, 2.1, 2.7], 'In': [2.3, 2.8, 3.4], 
            'Nb': [1.6, 1.9, 2.7], 'Ga': [2.3, 2.7, 3.4], 'Br': [2.1, 2.5, 3.0], 
            'Ir': [2.0, 2.3, 3.8], 'Be': [2.2, 2.7, 3.4], 
            'W': [1.7, 1.9, 2.2], 'Mg': [2.7, 3.2, 3.9], 'Sb': [2.2, 2.7, 3.3], 
            'Re': [1.9, 2.2, 2.8], 'Ba': [3.2, 3.9, 4.7], 
            'Rb': [3.9, 4.7, 5.5], 'Ag': [2.3, 2.7, 3.2], 'Hg': [2.7, 3.5, 4.3], 
            'Zn': [2.5, 3.2, 3.8], 'Cr': [1.5, 1.8, 2.3], 
            'Os': [1.9, 2.2, 2.8], 'Na': [2.8, 3.4, 4.1], 'H': [0.7, 0.9, 1.3], 
            'Sc': [2.0, 2.5, 3.1], 'Zr': [2.1, 2.5, 3.1], 
            'Se': [2.1, 2.3, 2.7], 'Al': [2.3, 2.8, 3.4], 'Rh': [2.0, 2.3, 2.7], 
            'Y': [2.4, 2.9, 3.6], 'B': [1.2, 1.5, 2.1], 
            'Ca': [2.8, 3.6, 4.6], 'Fe': [1.6, 2.0, 2.9], 'Tc': [1.5, 1.8, 2.2], 
            'Cs': [4.3, 5.0, 5.8], 'Ne': [2.0, 2.7, 3.3], 
            'C': [1.1, 1.4, 2.1], 'Ar': [2.8, 3.2, 3.7], 'He': [1.5, 2.0, 2.6], 
            'N': [0.9, 1.2, 1.6], 'Au': [2.3, 2.7, 3.2], 
            'Pt': [2.2, 2.6, 3.2], 'F': [1.3, 1.6, 2.1], 'Ge': [1.9, 2.2, 2.8], 
            'Co': [2.0, 2.4, 2.9], 'Cl': [1.6, 1.8, 2.2], 
            'Ti': [1.7, 2.2, 2.9], 'K': [3.0, 3.8, 4.6], 'V': [1.6, 1.9, 2.6], 
            'Cu': [2.0, 2.4, 3.0], 'Pb': [2.3, 2.7, 3.3], 
            'O': [1.1, 1.4, 2.1], 'As': [2.0, 2.3, 2.7], 'Li': [1.9, 2.4, 3.3], 
            'Bi': [2.4, 2.9, 3.5], 'Ru': [1.8, 2.1, 2.7], 
            'Sr': [3.5, 4.1, 4.7], 'Kr': [3.3, 4.0, 4.7], 'I': [2.4, 2.9, 3.5], 
            'Ta': [1.7, 2.0, 2.3], 'Mn': [1.5, 1.8, 2.5], 
            'Tl': [2.4, 3.3, 4.3], 'Ni': [1.9, 2.3, 2.8], 'P': [1.7, 2.2, 2.8], 
            'Hf': [2.3, 2.8, 3.4], 'Cd': [2.7, 3.6, 4.5]
        }
        return __data.get(elem, None)
    
class TestAtomSpecies(unittest.TestCase):
    
    from os.path import dirname as dname
    from os.path import join as pjoin
    fpsp = pjoin(dname(dname(dname(__file__))), 
        'tests', 'pporb', 'Si_ONCV_PBE-1.0.upf')
    def test_instantiation(self):
        # test the default case
        a = AtomSpecies('Si', 28, self.fpsp)
        self.assertEqual(a.elem, 'Si')
        self.assertEqual(a.mass, 28)
        self.assertEqual(a.fpsp, self.fpsp)
        self.assertEqual(a.forb, None)
        self.assertEqual(a.ecutjy, 100)
        self.assertEqual(a.rcutjy, 10)
        self.assertEqual(a.lmax, 3)
        self.assertEqual(a.orbgen, False)
        
        # test the FileNotFoundError for the fpsp
        with self.assertRaises(FileNotFoundError):
            a = AtomSpecies('Si', 28, 'nonexistent_file')
        
        # test full instantiation
        a = AtomSpecies('Si', 28, self.fpsp, None, 60, 6, 2, True)
        self.assertEqual(a.elem, 'Si')
        self.assertEqual(a.mass, 28)
        self.assertEqual(a.fpsp, self.fpsp)
        self.assertEqual(a.forb, None)
        self.assertEqual(a.ecutjy, 60)
        self.assertEqual(a.rcutjy, 6)
        self.assertEqual(a.lmax, 2)
        self.assertEqual(a.orbgen, True)
        
    def test_jygen(self):
        import os
        import uuid # generate temporary folder name to avoid accidental overwrite
        import shutil # will do the deletion
        a = AtomSpecies('Si', 28, self.fpsp)
        folder = str(uuid.uuid4())
        fn = a.jygen(folder)
        self.assertTrue(os.path.exists(fn))
        shutil.rmtree(folder)
    
    def test_aufbau(self):
        out = AtomSpecies.aufbau(10)
        self.assertEqual(out, {'1s': 2, 
                               '2s': 2, '2p': 6})
        
        out = AtomSpecies.aufbau(21) # Scandium
        self.assertEqual(out, {'1s': 2, 
                               '2s': 2, '2p': 6, 
                               '3s': 2, '3p': 6, 
                               '4s': 2, '3d': 1})
        
    def test_cal_nelec_that_can_fill(self):
        # Gallium: 1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p1, zval = 13
        # fill 4s, 3d, 4p
        out = AtomSpecies.cal_nelec_that_can_fill('Ga', [1, 1, 1, 0, 0], 13)
        self.assertEqual(out, 2 + 10 + 6)
        # +fill 5s, 4d, 5p, 4f (6s implicitly included)
        out = AtomSpecies.cal_nelec_that_can_fill('Ga', [2, 2, 2, 1, 0], 13)
        self.assertEqual(out, 18 + 2 + 10 + 6 + 14 + 2)
        # +fill 5d, 6p, 7s, 5f, 5g (6d, 7p, 8s implicitly included)
        out = AtomSpecies.cal_nelec_that_can_fill('Ga', [3, 3, 3, 2, 1], 13)
        self.assertEqual(out, 52 + 10 + 6 + 2 + 14 + 18 + 10 + 6 + 2)
        out = AtomSpecies.cal_nelec_that_can_fill('H', [5, 4, 3, 2, 1])
        self.assertEqual(out, 137)

    def test_slater_screening(self):
        self.assertAlmostEqual(
            AtomSpecies.cal_slater_screening_coef('H', 1, 0), 
            0, delta=1e-10)
        self.assertAlmostEqual(
            AtomSpecies.cal_slater_screening_coef('He', 1, 0), 
            0.3, delta=1e-10)
        self.assertAlmostEqual(
            AtomSpecies.cal_slater_screening_coef('F', 2, 1), 
            3.8, delta=1e-10)
        self.assertAlmostEqual(
            AtomSpecies.cal_slater_screening_coef('Ca', 4, 0), 
            17.15, delta=1e-10)
        self.assertAlmostEqual(
            AtomSpecies.cal_slater_screening_coef('Sc', 4, 0), 
            18, delta=1e-10)
        self.assertAlmostEqual(
            AtomSpecies.cal_slater_screening_coef('Cu', 4, 0), 
            25.3, delta=1e-10)
        self.assertAlmostEqual(
            AtomSpecies.cal_slater_screening_coef('Cu', 3, 2), 
            21.15, delta=1e-10)
        self.assertAlmostEqual(
            AtomSpecies.cal_slater_screening_coef('Pt', 6, 0), 
            74.45, delta=1e-10)
        self.assertAlmostEqual(
            AtomSpecies.cal_slater_screening_coef('Pt', 5, 1), 
            57.65, delta=1e-10)
        self.assertAlmostEqual(
            AtomSpecies.cal_slater_screening_coef('Os', 1, 0), 
            0.3, delta=1e-10)

    def test_build_hydrogen_orb(self):
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.integrate import simps
        dr = 0.01
        r = np.arange(0, 1001) * dr
        # s orbitals
        chi = [AtomSpecies.build_hydrogen_orb(
            elem='Kr', n=n, l=1, r=r, slater=False) 
            for n in range(2, 5)]
        for c in chi:
            plt.plot(r, c**2*r**2)
        plt.xlim(0, 4)
        plt.savefig('orbitals.png')
        plt.close()
        # check the orthogonality
        S = [[simps(c1*c2*r**2, r) for c2 in chi] for c1 in chi]
        S = np.array(S)
        self.assertTrue(np.allclose(S, np.eye(3), atol=1e-5))

if __name__ == '__main__':
    unittest.main()
