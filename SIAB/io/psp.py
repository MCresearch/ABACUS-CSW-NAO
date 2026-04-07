'''
read parts of pseudopotential
'''
# in-built modules
import os
import re
import unittest

# local modules
from SIAB.driver.control import OrbgenAssert

class PspParser:
    '''parser of pseudopotential file'''
    def __init__(self, fn):
        '''initialize the parser'''
        OrbgenAssert(os.path.exists(fn), 
                     f'{fn} not found',
                     FileNotFoundError)
        with open(fn, 'r') as f:
            self.data = f.readlines()
            # '\n' is removed
    
    @staticmethod
    def cut(data, item, mode='section'):
        '''cut data between start and end
        
        Parameters
        ----------
        data: list[str]|str
            the data to be cut
        item: str
            the item to be cut. For example: `PP_LOCAL`, ...
        mode: str
            the mode to cut the data. `section` or `tag`
        '''
        if mode == 'section': # <section> ... </section>
            regex = f'<{item}.*?>.*?</{item}>'
        elif mode == 'tag':  # <tag .../>
            regex = f'<{item}.*?/>'
        else:
            OrbgenAssert(False, f'Unknown mode: {mode}', ValueError)
        data = '\n'.join(data) if isinstance(data, list) else data
        return re.findall(regex, data, re.DOTALL)
    
    def vlocal(self):
        '''read local potential'''
        data = PspParser.cut(self.data, 'PP_LOCAL')[0]
        data = re.split(r'</?PP_LOCAL.*>', data)[1]
        return [float(x) for x in data.split()]
    
    def r(self):
        '''read radial grid'''
        data = PspParser.cut(self.data, 'PP_R')[0]
        data = re.split(r'</?PP_R.*>', data)[1]
        return [float(x) for x in data.split()]
    
    def zval(self):
        '''read the zval from PP_HEADER'''
        data = PspParser.cut(self.data, 'PP_HEADER', mode='tag')[0]
        return float(re.search(r'z_valence=\s*"\s*([0-9.]+)', data).group(1))
    
class TestPspParser(unittest.TestCase):
    
    here = os.path.dirname(__file__)
    root = os.path.dirname(here)
    root = os.path.dirname(root)
    fn = os.path.join(root, 'tests', 'pporb', 'Si_ONCV_PBE-1.0.upf')
    
    def test_init(self):
        with self.assertRaises(FileNotFoundError):
            _ = PspParser('non_exist_file')
        pspparser = PspParser(self.fn)
        self.assertFalse(pspparser.data == '')

    def test_cut(self):
        with open(self.fn, 'r') as f:
            data = f.read()
        section = PspParser.cut(data, 'PP_LOCAL')
        self.assertTrue(len(section) != 0)
        tag = PspParser.cut(data, 'PP_HEADER', mode='tag')
        self.assertTrue(len(tag) != 0)

    def test_vlocal(self):
        pspparser = PspParser(self.fn)
        vlocal = pspparser.vlocal()
        self.assertTrue(len(vlocal) == 602)
        self.assertTrue(all(isinstance(x, float) for x in vlocal))

    def test_r(self):
        pspparser = PspParser(self.fn)
        r = pspparser.r()
        self.assertTrue(len(r) == 602)
        self.assertTrue(all(isinstance(x, float) for x in r))
        # r is always in an ascending order
        self.assertTrue(all(r[i] < r[i+1] for i in range(len(r)-1)))

    def test_zval(self):
        pspparser = PspParser(self.fn)
        zval = pspparser.zval()
        self.assertTrue(isinstance(zval, float))
        self.assertTrue(zval == 4.0)

if __name__ == '__main__':
    unittest.main()