import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Batch, Data
from rdkit import Chem
from tqdm import tqdm

from ..log import get_logger
from .args import DiffMSArgs
from .dataset import DiffMSDatasetInfo
from .graph_transformer import GraphTransformer
from .noise_schedule import PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from .utils import GraphData, to_dense
from .metrics import DiffMSMetrics, TrainLossDiscrete, TrainMetrics
from .diffusion_utils import mask_distributions, sum_except_batch, posterior_distributions, \
    sample_discrete_features, sample_discrete_feature_noise, compute_batched_over0_posterior_distribution

logger = get_logger(__file__)

class DiffMS(pl.LightningModule):
    def __init__(
        self,
        args: DiffMSArgs,
        dataset_infos: DiffMSDatasetInfo,
        visualization_tools,
        extra_features,
        domain_features
    ):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims

        self.args = args
        self.name = args.experiment_name
        self.T = args.diffusion_steps

        self.dims = {
            'X': (input_dims['X'], output_dims['X']),
            'E': (input_dims['E'], output_dims['E']),
            'y': (input_dims['y'], output_dims['y']),
        }
        self.node_dist = dataset_infos.nodes_dist

        self.val_num_samples = args.samples_to_generate
        self.test_num_samples = args.samples_to_generate
        self.train_loss = TrainLossDiscrete(args.lambda_train)
        self.train_metrics = TrainMetrics()
        self.val_metrics = DiffMSMetrics(num_samples=self.val_num_samples, split='val')
        self.test_metrics = DiffMSMetrics(num_samples=self.test_num_samples, split='test')

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model = GraphTransformer(
            n_layers=args.n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=args.hidden_mlp_dims,
            hidden_dims=args.hidden_dims,
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU()
        )
        
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            args.diffusion_noise_schedule,
            timesteps=args.diffusion_steps
        )
        self.denoise_nodes = args.denoise_nodes
        
        node_types = dataset_infos.node_types.float()
        x_marginals = node_types / torch.sum(node_types)
        edge_types = dataset_infos.edge_types.float()
        e_marginals = edge_types / torch.sum(edge_types)
        
        self.transition_model = MarginalUniformTransition(
            x_marginals=x_marginals,
            e_marginals=e_marginals,
            y_classes=self.dims['y'][1]
        )
        self.limit_dist = GraphData(
            X=x_marginals,
            E=e_marginals,
            y=torch.ones(self.dims['y'][1]) / self.dims['y'][1],
        )

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.log_every_steps = args.log_every_steps
        self.best_val_nll = 1e8

        logging.info(f"Finished initializing DiffMS on GPU {self.device}")

    def process(self, data: Data):
        dense_data, node_mask = to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        return dense_data, noisy_data, extra_data, node_mask

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.args.lr, amsgrad=True, weight_decay=self.args.weight_decay)
        stepping_batches = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.args.lr, total_steps=stepping_batches, pct_start=self.args.pct_start)
        lr_scheduler = {
            'scheduler': scheduler,
            'name': 'learning_rate',
            'interval':'step',
            'frequency': 1,
        }
        return [opt], [lr_scheduler]
        
    def on_train_epoch_start(self) -> None:
        logger.info(f'Training epoch {self.current_epoch} started')
        self.train_loss.reset()
        self.train_metrics.reset()
    
    def training_step(self, data: Data, batch_idx: int):
        dense_data, noisy_data, extra_data, node_mask = self.process(data)
        gt = GraphData(dense_data.X, dense_data.E, data.y)
        pred = self.forward(noisy_data, extra_data, node_mask)
        loss, log_dict = self.train_loss(pred, gt)
        self.log_dict(log_dict, sync_dist=True)
        metrics = self.train_metrics(pred, gt)
        self.log_dict(metrics, sync_dist=True)
        return {'loss': loss}
    
    def predict_forcing(self, pred: GraphData, gt: GraphData):
        pred.X = gt.X
        pred.y = gt.y
        return pred

    def on_validation_epoch_start(self) -> None:
        self.val_metrics.reset()

    def validation_step(self, data: Data, batch_idx: int):
        dense_data, noisy_data, extra_data, node_mask = self.process(data)
        gt = GraphData(dense_data.X, dense_data.E, data.y)
        pred = self.forward(noisy_data, extra_data, node_mask)
        pred = self.predict_forcing(pred, gt)
        
        nll = self.compute_val_loss(pred, noisy_data, gt,  node_mask, test=False)
        true_E = torch.reshape(dense_data.E, (-1, dense_data.E.size(-1)))  # (bs * n * n, de)
        masked_pred_E = torch.reshape(pred.E, (-1, pred.E.size(-1)))   # (bs * n * n, de)
        mask_E = (true_E != 0.).any(dim=-1)
        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]
        self.val_metrics.cross_entropy(flat_pred_E, flat_true_E)
        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()    
        self.log_dict(metrics, sync_dist=True)
        # TODO: add checkpointing

    def on_test_epoch_start(self) -> None:
        self.test_metrics.reset()

    def test_step(self, data: Data, batch_idx: int):
        dense_data, noisy_data, extra_data, node_mask = self.process(data)
        gt = GraphData(X=dense_data.X, E=dense_data.E, y=data.y)
        pred = self.forward(noisy_data, extra_data, node_mask)
        pred = self.predict_forcing(pred, gt)
        nll = self.compute_val_loss(pred, noisy_data, gt, node_mask, test=True)

        true_E = torch.reshape(gt.E, (-1, gt.E.size(-1)))  # (bs * n * n, de)
        masked_pred_E = torch.reshape(pred.E, (-1, pred.E.size(-1)))   # (bs * n * n, de)
        mask_E = (true_E != 0.).any(dim=-1)
        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]
        self.test_metrics.cross_entropy(flat_pred_E, flat_true_E)

        true_mols = [Chem.MolFromSmiles(data.get_example(idx).smiles) for idx in range(len(data))] # TODO: check
        predicted_mols = [list() for _ in range(len(data))]
        for _ in range(self.test_num_samples):
            for idx, mol in enumerate(self.sample_batch(data)):
                predicted_mols[idx].append(mol)
        
        for idx in range(len(data)):
            self.test_metrics.k_acc.update(predicted_mols[idx], true_mols[idx])
            self.test_metrics.sim_metrics.update(predicted_mols[idx], true_mols[idx])
            self.test_metrics.validity.update(predicted_mols[idx])

        return {'loss': nll}

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, sync_dist=True)        

    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = mask_distributions(
            true_X=limit_X.clone(),
            true_E=limit_E.clone(),
            pred_X=probX,
            pred_E=probE,
            node_mask=node_mask
        )

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')
        return sum_except_batch(kl_distance_X) + sum_except_batch(kl_distance_E)

    def compute_Lt(self, gt: GraphData, pred: GraphData, noisy_data, node_mask, test):
        X, E, y = gt.X, gt.E, gt.y
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = posterior_distributions(
            X=X, E=E,
            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
            Qt=Qt, Qsb=Qsb, Qtb=Qtb
        )
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = posterior_distributions(
            X=pred_probs_X, E=pred_probs_E,
            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
            Qt=Qt, Qsb=Qsb, Qtb=Qtb
        )
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true.X, prob_true.E, prob_pred.X, prob_pred.E = mask_distributions(
            true_X=prob_true.X,
            true_E=prob_true.E,
            pred_X=prob_pred.X,
            pred_E=prob_pred.E,
            node_mask=node_mask
        )
        if test:
            kl_x = self.test_metrics.X_kl(prob_true.X, torch.log(prob_pred.X))
            kl_e = self.test_metrics.E_kl(prob_true.E, torch.log(prob_pred.E))
        else:
            kl_x = self.val_metrics.X_kl(prob_true.X, torch.log(prob_pred.X))
            kl_e = self.val_metrics.E_kl(prob_true.E, torch.log(prob_pred.E))
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, y, node_mask):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.dims['X'][1]).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.dims['E'][1]).float()
        y0 = y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = GraphData(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {
            'X_t': sampled_0.X,
            'E_t': sampled_0.E,
            'y_t': sampled_0.y,
            'node_mask': node_mask,
            't': torch.zeros(X0.shape[0], 1).type_as(y0)
        }
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.dims['X'][1]).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.dims['E'][1]).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.dims['E'][1]).type_as(probE0)

        return GraphData(X=probX0, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        lowest_t = 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = X
        if self.denoise_nodes:
            X_t = F.one_hot(sampled_t.X, num_classes=self.dims['X'][1])
        E_t = F.one_hot(sampled_t.E, num_classes=self.dims['E'][1])
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = GraphData(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def compute_val_loss(self, pred: GraphData, noisy_data: dict, gt: GraphData, node_mask, test=False):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
        """
        X, E, y = gt.X, gt.E, gt.y
        t = noisy_data['t']

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(gt, pred, noisy_data, node_mask, test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, y, node_mask)

        loss_term_0 = self.val_metrics.X_logp(X * prob0.X.log()) + self.val_metrics.E_logp(E * prob0.E.log())

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        nll = self.test_metrics.nll(nlls) if test else self.val_metrics.nll(nlls)

        return nll

    def forward(self, noisy_data, extra_data, node_mask) -> GraphData:
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)
    
    @torch.no_grad()
    def sample_batch(self, batch: Batch) -> list[Chem.Mol]:
        dense_data, node_mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        z_T = sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = dense_data.X, z_T.E, batch.y

        assert (E == torch.transpose(E, 1, 2)).all()

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((len(batch), 1), dtype=torch.float32, device=self.device)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, __ = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask)
            _, E, y = sampled_s.X, sampled_s.E, batch.y

        # Sample
        sampled_s.X = X
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, batch.y

        mols = []

        for nodes, adj_mat in zip(X, E):
            mols.append(self.visualization_tools.mol_from_graphs(nodes, adj_mat))

        return mols

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        p_s_and_t_given_0_X = compute_batched_over0_posterior_distribution(
            X_t=X_t,
            Qt=Qt.X,
            Qsb=Qsb.X,
            Qtb=Qtb.X
        )

        p_s_and_t_given_0_E = compute_batched_over0_posterior_distribution(
            X_t=E_t,
            Qt=Qt.E,
            Qsb=Qsb.E,
            Qtb=Qtb.E
        )
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.dims['X'][1]).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.dims['E'][1]).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = GraphData(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = GraphData(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return GraphData(X=extra_X, E=extra_E, y=extra_y)