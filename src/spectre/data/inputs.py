import os
from typing import Iterable, Dict

import torch
import torch.nn.functional as F

from ..core.const import INPUT_TYPES, UNK_IDX, FORMULA_RE, ELEM2IDX
from .fp_loader import FPLoader

def parse_formula(formula: str) -> dict[str,int]:
    """
    Turn "C20H25BrN2O2" → {"C":20, "H":25, "Br":1, "N":2, "O":2}
    """
    counts: dict[str,int] = {}
    for elem, cnt in FORMULA_RE.findall(formula):
        counts[elem] = int(cnt) if cnt else 1
    return counts

class SpectralInputLoader:
    '''
    Represents the SPECTRE input data types.
    
    - HSQC NMR ('hsqc')
    - H NMR ('h_nmr')
    - C NMR ('c_nmr')
    - MS/MS ('mass_spec')
    - Molecular Weight ('mw')
    - Chemical Formula ('formula')
    '''
    def __init__(self, root: str, data_dict: dict, dtype=torch.float32):
        '''
        In index.pkl, it is stored idx: data_dict pairs. Feed this in for initialization.
        '''
        self.root = root
        self.data_dict = data_dict
        self.dtype = dtype
        
        # if you ever end up storing non-cpu tensors, you can add 'map_location': 'cpu'
        # if old pytorch, you may need to remove weights_only = True
        self.kwargs = {'weights_only': True} 
    
    def load(self, idx, input_types: Iterable[INPUT_TYPES], jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        '''
        Load spectral inputs.
        
        Assumptions: each item in input_types is valid in this input entry.
        
        - root: the dataset root.
        - input_types: an iterable of input types to include.
        - jittering: if greater than 0, apply jittering to input spectra.
        
        Returns:
        Dictionary of requested input types and their data
        '''
        
        data_inputs = {}
        for input_type in input_types:
            data_inputs.update(getattr(self, f'_load_{input_type}')(idx, jittering))
        return data_inputs

    def _load_hsqc(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        filename = f'{idx}.pt'
        hsqc: torch.Tensor = torch.load(os.path.join(self.root, 'HSQC_NMR', filename), **self.kwargs)
        hsqc = hsqc.to(dtype=self.dtype)
        if jittering > 0:
            hsqc[:,0] = hsqc[:,0] + torch.randn_like(hsqc[:,0]) * jittering
            hsqc[:,1] = hsqc[:,1] + torch.randn_like(hsqc[:,1]) * jittering * 0.1
        return {'hsqc': hsqc}
    
    def _load_c_nmr(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        filename = f'{idx}.pt'
        c_nmr: torch.Tensor = torch.load(os.path.join(self.root, 'C_NMR', filename), **self.kwargs)
        c_nmr = c_nmr.to(dtype=self.dtype)
        c_nmr = c_nmr.view(-1,1)                   # (N,1)
        c_nmr = F.pad(c_nmr, (0,2), "constant", 0) # -> (N,3)
        if jittering > 0:
            c_nmr = c_nmr + torch.randn_like(c_nmr) * jittering
        return {'c_nmr': c_nmr}
    
    def _load_h_nmr(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        filename = f'{idx}.pt'
        h_nmr: torch.Tensor = torch.load(os.path.join(self.root, 'H_NMR', filename), **self.kwargs)
        h_nmr = h_nmr.to(dtype=self.dtype)
        h_nmr = h_nmr.view(-1,1)                    # (N,1)
        h_nmr = F.pad(h_nmr, (1,1), "constant", 0)  # -> (N,3)
        if jittering > 0:
            h_nmr = h_nmr + torch.randn_like(h_nmr) * jittering * 0.1
        return {'h_nmr': h_nmr}
    
    def _load_mass_spec(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        filename = f'{idx}.pt'
        mass_spec: torch.Tensor = torch.load(os.path.join(self.root, 'MassSpec', filename), **self.kwargs)
        mass_spec = mass_spec.to(dtype=self.dtype)
        mass_spec = F.pad(mass_spec, (0,1), "constant", 0)
        if jittering > 0:
            noise = torch.zeros_like(mass_spec)
            noise[:, 0] = torch.randn_like(mass_spec[:, 0]) * mass_spec[:, 0] / 100_000  # jitter m/z
            noise[:, 1] = torch.randn_like(mass_spec[:, 1]) * mass_spec[:, 1] / 10
            mass_spec = mass_spec + noise
        return {'mass_spec': mass_spec}
    
    def _load_mw(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        return {'mw': torch.tensor(self.data_dict[idx]['mw'], dtype=self.dtype)}
    
    def _load_formula(self, idx: int, jittering: float = 0.0) -> Dict[str, torch.Tensor]:
        formula = self.data_dict[idx]['formula']
        elem_counts = parse_formula(formula)
        ordered = []
        if 'C' in elem_counts: ordered.append('C')
        if 'H' in elem_counts: ordered.append('H')
        for e in sorted(e for e in elem_counts if e not in ('C','H')):
            ordered.append(e)
        idxs = [ELEM2IDX.get(e, UNK_IDX) for e in ordered]
        cnts = [elem_counts[e] for e in ordered]
        return {
            'elem_idx': torch.tensor(idxs, dtype=torch.long),
            'elem_cnt': torch.tensor(cnts, dtype=torch.long)
        }

class MFInputLoader:
    '''
    The Morgan Fingerprint groundtruth loader.
    '''
    def __init__(self, fp_loader: FPLoader):
        self.fp_loader = fp_loader
    
    def load(self, idx: int) -> torch.Tensor:
        return self.fp_loader.build_mfp(idx)