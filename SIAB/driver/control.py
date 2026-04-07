# in-built modules
import unittest
import logging

def OrbgenAssert(cond, f, errtyp=ValueError):
    '''
    assert that the cond holds

    Parameters
    ----------
    cond: bool
        the condition to be asserted
    f: str
        the error message, or a callable function
    errtyp: Exception
        the type of exception to be raised
    '''
    if not cond:
        err = f() if not isinstance(f, str) else f
        logging.error(err)
        raise errtyp(err)
    
def OrbgenAssertIn(arr, domain, f, errtyp=ValueError):
    '''
    assert that ALL elements in arr are in domain

    Parameters
    ----------
    arr: list
        the list to be asserted
    domain: list
        the domain of the list
    f: callable
        the error message
    errtyp: Exception
        the type of exception to be raised
    '''
    for a in arr:
        if a not in domain:
            err = f(a) if not isinstance(f, str) else f
            logging.error(err)
            raise errtyp(err)

class TestDriverControl(unittest.TestCase):

    def test_OrbgenAssert(self):
        OrbgenAssert(True, 'test_OrbgenAssert')
        with self.assertRaises(ValueError):
            OrbgenAssert(False, 'test_OrbgenAssert')

    def test_OrbgenAssertIn(self):
        OrbgenAssertIn([1, 2], [1, 2, 3], lambda x: f'{x} not in domain')
        with self.assertRaises(ValueError):
            OrbgenAssertIn([1, 2, 4], [1, 2, 3], lambda x: f'{x} not in domain')

if __name__ == '__main__':
    unittest.main()
