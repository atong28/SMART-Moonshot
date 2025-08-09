import re
from rdkit import Chem

DEBUG_LEN = 3000

DROP_MW_PERCENTAGE = 0.5
DROP_MS_PERCENTAGE = 0.5
DROP_FORMULA_PERCENTAGE = 0.5

INPUTS_CANONICAL_ORDER = ['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw', 'formula']

_pt = Chem.GetPeriodicTable()

ELEMENT_VOCAB = [_pt.GetElementSymbol(Z) for Z in range(1, 119)]
# reserve 0 for PAD, and optionally len+1 for UNK
ELEM2IDX = {sym: i+1 for i, sym in enumerate(ELEMENT_VOCAB)}
UNK_IDX = len(ELEMENT_VOCAB) + 1

# regex to split element symbols (1 or 2 letters) and optional count
FORMULA_RE = re.compile(r'([A-Z][a-z]?)(\d*)')

DO_NOT_OVERRIDE = ['train', 'test', 'visualize', 'load_from_checkpoint']