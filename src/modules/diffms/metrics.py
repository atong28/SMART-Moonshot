import torch
import torch.nn as nn
from torchmetrics import Metric, MetricCollection
from collections import Counter
from typing import List
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import torch.nn.functional as F
from .utils import is_valid, canonical_mol_from_inchi, GraphData
from ..core.const import ATOM_DECODER

class CEPerClass(Metric):
    full_state_update = False
    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.softmax = torch.nn.Softmax(dim=-1)
        self.binary_cross_entropy = torch.nn.BCELoss(reduction='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model   (bs, n, d) or (bs, n, n, d)
            target: Ground truth values     (bs, n, d) or (bs, n, n, d)
        """
        target = target.reshape(-1, target.shape[-1])
        mask = (target != 0.).any(dim=-1)

        prob = self.softmax(preds)[..., self.class_id]
        prob = prob.flatten()[mask]

        target = target[:, self.class_id]
        target = target[mask]

        output = self.binary_cross_entropy(prob, target)
        self.total_ce += output
        self.total_samples += prob.numel()

    def compute(self):
        return self.total_ce / self.total_samples


class HydrogenCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class CarbonCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NitroCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class OxyCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class FluorCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BoronCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BrCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class ClCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class IodineCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class PhosphorusCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SulfurCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SeCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SiCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NoBondCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SingleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class DoubleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class TripleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AromaticCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AtomMetricsCE(MetricCollection):
    def __init__(self):
        class_dict = {
            'H': HydrogenCE, 'C': CarbonCE, 'N': NitroCE, 'O': OxyCE, 'F': FluorCE, 'B': BoronCE,
            'Br': BrCE, 'Cl': ClCE, 'I': IodineCE, 'P': PhosphorusCE, 'S': SulfurCE, 'Se': SeCE,
            'Si': SiCE
        }

        metrics_list = []
        for i, atom_type in enumerate(ATOM_DECODER):
            try:
                metrics_list.append(class_dict[atom_type](i))
            except KeyError:
                pass
        super().__init__(metrics_list)


class BondMetricsCE(MetricCollection):
    def __init__(self):
        ce_no_bond = NoBondCE(0)
        ce_SI = SingleCE(1)
        ce_DO = DoubleCE(2)
        ce_TR = TripleCE(3)
        ce_AR = AromaticCE(4)
        super().__init__([ce_no_bond, ce_SI, ce_DO, ce_TR, ce_AR])


class TrainMetrics(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_atom_metrics = AtomMetricsCE()
        self.train_bond_metrics = BondMetricsCE()

    def forward(self, pred: GraphData, gt: GraphData):
        self.train_atom_metrics(pred.X, gt.X)
        self.train_bond_metrics(pred.E, gt.E)
        metrics = {}
        for key, val in self.train_atom_metrics.compute().items():
            metrics['train/' + key] = val.item()
        for key, val in self.train_bond_metrics.compute().items():
            metrics['train/' + key] = val.item()

        self.reset()
        return metrics

    def reset(self):
        self.train_atom_metrics.reset()
        self.train_bond_metrics.reset()

class K_ACC(Metric):
    def __init__(self, k: int, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.k = k
        self.add_state("correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, generated_inchis: List[str], true_inchi: str):
        if true_inchi in generated_inchis[: self.k]:
            self.correct += 1
        self.total += 1

    def compute(self) -> torch.Tensor:
        """Compute final top-k accuracy."""
        if self.total == 0:
            return torch.tensor(0.0, device=self.correct.device)
        return self.correct.float() / self.total.float()


class K_ACC_Collection(Metric):
    """
    A collection of K_ACC metrics for multiple values of k.
    """
    def __init__(self, k_list: List[int], dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.metrics = nn.ModuleDict()
        for k in k_list:
            self.metrics[f"acc_at_{k}"] = K_ACC(k, dist_sync_on_step=dist_sync_on_step)

    def update(self, generated_mols: List[Chem.Mol], true_mol: Chem.Mol):
        # Filter out invalid molecules, and select unique InChIs by frequency
        inchis = [Chem.MolToInchi(mol) for mol in generated_mols if is_valid(mol)]
        inchi_counter = Counter(inchis)
        # Sort by frequency, keep unique
        inchis = [item for item, _count in inchi_counter.most_common()]
        true_inchi = Chem.MolToInchi(true_mol)

        # Update each K_ACC submetric
        for metric in self.metrics.values():
            metric.update(inchis, true_inchi)

    def compute(self):
        return {name: m.compute() for name, m in self.metrics.items()}

class K_TanimotoSimilarity(Metric):
    def __init__(self, k: int, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.k = k
        self.add_state("similarity_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, generated_mols: List[Chem.Mol], true_mol: Chem.Mol):
        true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2, nBits=2048)
        max_sim = 0.0
        for mol in generated_mols[: self.k]:
            try:
                gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                sim = DataStructs.TanimotoSimilarity(gen_fp, true_fp)
                max_sim = max(max_sim, sim)
            except Exception:
                pass
        self.similarity_sum += max_sim
        self.total += 1

    def compute(self) -> torch.Tensor:
        """Compute the average max Tanimoto similarity."""
        if self.total == 0:
            return torch.tensor(0.0, device=self.similarity_sum.device)
        return self.similarity_sum / self.total.float()


class K_CosineSimilarity(Metric):
    def __init__(self, k: int, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.k = k
        self.add_state("similarity_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, generated_mols: List[Chem.Mol], true_mol: Chem.Mol):
        true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2, nBits=2048)
        max_sim = 0.0
        for mol in generated_mols[: self.k]:
            try:
                gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                sim = DataStructs.CosineSimilarity(gen_fp, true_fp)
                max_sim = max(max_sim, sim)
            except Exception:
                pass
        self.similarity_sum += max_sim
        self.total += 1

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0, device=self.similarity_sum.device)
        return self.similarity_sum / self.total.float()


class K_SimilarityCollection(Metric):
    def __init__(self, k_list: List[int], dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.metrics = nn.ModuleDict()
        for k in k_list:
            self.metrics[f"tanimoto_at_{k}"] = K_TanimotoSimilarity(k, dist_sync_on_step=dist_sync_on_step)
            self.metrics[f"cosine_at_{k}"] = K_CosineSimilarity(k, dist_sync_on_step=dist_sync_on_step)

    def update(self, generated_mols: List[Chem.Mol], true_mol: Chem.Mol):
        inchis = [Chem.MolToInchi(mol) for mol in generated_mols if is_valid(mol)]
        inchi_counter = Counter(inchis)
        inchis = [item for item, _count in inchi_counter.most_common()]

        processed_mols = [canonical_mol_from_inchi(inchi) for inchi in inchis]

        for metric in self.metrics.values():
            metric.update(processed_mols, true_mol)

    def compute(self):
        return {name: m.compute() for name, m in self.metrics.items()}


class Validity(Metric):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("valid", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, generated_mols: List[Chem.Mol]):
        for mol in generated_mols:
            if is_valid(mol):
                self.valid += 1
            self.total += 1

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0, device=self.valid.device)
        return self.valid.float() / self.total.float()

class SumExceptBatchKL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, p, q) -> None:
        self.total_value += F.kl_div(q, p, reduction='sum')
        self.total_samples += p.size(0)

    def compute(self):
        return self.total_value / self.total_samples
    
class CrossEntropyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """ Update state with predictions and targets.
            preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
            target: Ground truth values     (bs * n, d) or (bs * n * n, d). """
        target = torch.argmax(target, dim=-1)
        output = F.cross_entropy(preds, target, reduction='sum')
        self.total_ce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_ce / self.total_samples

class NLL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_nll', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, batch_nll) -> None:
        self.total_nll += torch.sum(batch_nll)
        self.total_samples += batch_nll.numel()

    def compute(self):
        return self.total_nll / self.total_samples

class SumExceptBatchMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, values) -> None:
        self.total_value += torch.sum(values)
        self.total_samples += values.shape[0]

    def compute(self):
        return self.total_value / self.total_samples

class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_train = lambda_train

    def forward(self, pred: GraphData, gt: GraphData):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean. """
        true_X = torch.reshape(gt.X, (-1, gt.X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(gt.E, (-1, gt.E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(pred.X, (-1, pred.X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(pred.E, (-1, pred.E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        loss_y = self.y_loss(pred.y, gt.y) if gt.y.numel() > 0 else 0.0

        to_log = {
            "train_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
            "train_loss/X_CE": self.node_loss.compute() if true_X.numel() > 0 else -1,
            "train_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1,
            "train_loss/y_CE": self.y_loss.compute() if gt.y.numel() > 0 else -1
        }
        self.reset()
        return self.lambda_train[0] * loss_X + self.lambda_train[1] * loss_E + self.lambda_train[2] * loss_y, to_log

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

class DiffMSMetrics(nn.Module):
    def __init__(self, num_samples: int, split: str):
        super().__init__()
        self.nll = NLL()
        self.X_kl = SumExceptBatchKL()
        self.E_kl = SumExceptBatchKL()
        self.X_logp = SumExceptBatchMetric()
        self.E_logp = SumExceptBatchMetric()
        self.k_acc = K_ACC_Collection(list(range(1, num_samples + 1)))
        self.sim_metrics = K_SimilarityCollection(list(range(1, num_samples + 1)))
        self.validity = Validity()
        self.cross_entropy = CrossEntropyMetric()
        self.split = split
        
    def reset(self):
        self.nll.reset()
        self.X_kl.reset()
        self.E_kl.reset()
        self.X_logp.reset()
        self.E_logp.reset()
        self.k_acc.reset()
        self.sim_metrics.reset()
        self.validity.reset()
        self.cross_entropy.reset()

    def compute(self):
        metrics = {
            f'{self.split}/NLL': self.nll.compute(),
            f'{self.split}/X_KL': self.X_kl.compute(),
            f'{self.split}/E_KL': self.E_kl.compute(),
            f'{self.split}/X_logp': self.X_logp.compute(),
            f'{self.split}/E_logp': self.E_logp.compute(),
            f'{self.split}/cross_entropy': self.cross_entropy.compute()
        }
        if self.split == 'test':
            for key, value in self.k_acc.compute().items():
                metrics[f"{self.split}/{key}"] = value
            for key, value in self.sim_metrics.compute().items():
                metrics[f"{self.split}/{key}"] = value
            metrics[f"{self.split}/validity"] = self.validity.compute()
        return metrics