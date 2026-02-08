import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .encoder import HSQCPeakEncoder

class HSQC2SelfiesModel(nn.Module):
    def __init__(self, vocab_size:int, pad_id:int, d_model=256, nhead=8, num_layers=4, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = pad_id

        self.hsqc = HSQCPeakEncoder(d_model=d_model, dropout=dropout)

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout, activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        

    def forward(self, peaks, peaks_mask, y_in):
        memory, _ = self.hsqc(peaks, peaks_mask)

        B, T = y_in.shape
        if T > self.max_len:
            raise ValueError(f"T={T} exceeds max_len={self.max_len}")

        pos = torch.arange(T, device=y_in.device).unsqueeze(0).expand(B, T)
        y = self.tok_emb(y_in) + self.pos_emb(pos)

        causal = torch.triu(torch.ones(T, T, device=y_in.device), diagonal=1).bool()

        out = self.decoder(
            tgt=y,
            memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=(y_in == self.pad_id),
            memory_key_padding_mask=~peaks_mask,
        )
        out = self.ln(out)
        return self.lm_head(out)

    @torch.no_grad()
    def generate(self, peaks, peaks_mask, bos_id, eos_id, max_new_tokens=512):
        self.eval()
        device = peaks.device
        B = peaks.shape[0]

        memory, _ = self.hsqc(peaks, peaks_mask)

        # Start with BOS
        y = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            T = y.shape[1]
            if T >= self.max_len:
                break

            pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            tgt = self.tok_emb(y) + self.pos_emb(pos)

            causal = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

            out = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=causal,
                tgt_key_padding_mask=(y == self.pad_id),
                memory_key_padding_mask=~peaks_mask,
            )
            logits = self.lm_head(self.ln(out))          # [B, T, V]
            next_id = logits[:, -1].argmax(dim=-1)       # [B]

            # If a sequence already finished, keep it "stopped"
            # Option A: keep appending PAD after EOS (clean for length metrics)
            next_id = torch.where(
                finished,
                torch.full_like(next_id, self.pad_id),
                next_id,
            )

            y = torch.cat([y, next_id.unsqueeze(1)], dim=1)

            # Mark newly finished sequences (only those not finished before)
            finished = finished | (next_id == eos_id)

            # Stop when all sequences have produced EOS at least once
            if finished.all():
                break

        return y


class LitHSQC2Selfies(pl.LightningModule):
    def __init__(self, model: HSQC2SelfiesModel, pad_id: int, bos_id: int, eos_id: int, lr=3e-4):
        super().__init__()
        self.model = model

        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.lr = lr

        self._greedy_eos_rate = []
        self._greedy_no_eos_rate = []
        self._greedy_len_ratio = []
        self._greedy_exact = []
        self._greedy_pos_ratio = []
        self._greedy_pos_diff = []
        self._greedy_tok_acc = []

    def training_step(self, batch, batch_idx):
        peaks, peaks_mask, y_in, y_out, y_mask = batch
        logits = self.model(peaks, peaks_mask, y_in)   # [B,T,V]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y_out.reshape(-1),
            ignore_index=self.pad_id,
        )

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct = ((pred == y_out) & y_mask).sum()
            total = y_mask.sum().clamp_min(1)
            acc = correct.float() / total.float()

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/token_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        peaks, peaks_mask, y_in, y_out, y_mask = batch

        logits = self.model(peaks, peaks_mask, y_in)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y_out.reshape(-1),
            ignore_index=self.pad_id,
        )
        
        self.log("val/loss", loss, prog_bar=True)

        # Greedy metrics for first K batches
        K = 5
        if batch_idx >= K or not self.trainer.is_global_zero:
            return

        with torch.no_grad():
            # IMPORTANT: move to device for generate
            peaks = peaks.to(self.device)
            peaks_mask = peaks_mask.to(self.device)
            y_out = y_out.to(self.device)

            gen = self.model.generate(
                peaks[:8], peaks_mask[:8],
                bos_id=self.bos_id, eos_id=self.eos_id,
                max_new_tokens=self.model.max_len,
            )[:, 1:]  # drop BOS

            tgt = y_out[:8]  # includes EOS in targets

            # core lengths + eos positions
            gen_len, gen_has, gen_eos_pos = self._eos_aware_len(gen, self.eos_id, self.pad_id)
            tgt_len, tgt_has, tgt_eos_pos = self._eos_aware_len(tgt, self.eos_id, self.pad_id)

            eos_rate = gen_has.float().mean()
            no_eos_rate = (~gen_has).float().mean()

            # metrics that require EOS in both
            mask = gen_has & tgt_has
            if mask.any():
                len_ratio = (gen_len[mask].float() / tgt_len[mask].float().clamp_min(1)).mean()
                eos_pos_diff = (gen_eos_pos[mask].float() - tgt_eos_pos[mask].float()).mean()
                eos_pos_ratio = (gen_eos_pos[mask].float() / tgt_eos_pos[mask].float().clamp_min(1)).mean()
            else:
                len_ratio = torch.tensor(float("nan"), device=self.device)
                eos_pos_diff = torch.tensor(float("nan"), device=self.device)
                eos_pos_ratio = torch.tensor(float("nan"), device=self.device)

            exact_mean, exact_mask, *_ = self._exact_match_eos_aware(gen, tgt, self.eos_id, self.pad_id)

            # optional token accuracy up to min length (EOS-aware) on masked items
            if mask.any():
                maxT = min(gen.size(1), tgt.size(1))
                # compare only up to maxT, but don’t count PADs in tgt
                eq = (gen[:, :maxT] == tgt[:, :maxT])
                tgt_not_pad = (tgt[:, :maxT] != self.pad_id)
                tok_acc = (eq & tgt_not_pad)[mask].float().sum() / tgt_not_pad[mask].float().sum().clamp_min(1)
            else:
                tok_acc = torch.tensor(float("nan"), device=self.device)

            # stash per-batch so epoch_end can average
            self._greedy_eos_rate.append(eos_rate.detach().cpu())
            self._greedy_no_eos_rate.append(no_eos_rate.detach().cpu())
            self._greedy_len_ratio.append(len_ratio.detach().cpu())
            self._greedy_pos_diff.append(eos_pos_diff.detach().cpu())
            self._greedy_pos_ratio.append(eos_pos_ratio.detach().cpu())
            self._greedy_exact.append(exact_mean.detach().cpu())
            self._greedy_tok_acc.append(tok_acc.detach().cpu())

    def on_validation_start(self):
        if not self.trainer.is_global_zero:
            return
        self._greedy_eos_rate = []
        self._greedy_no_eos_rate = []
        self._greedy_len_ratio = []
        self._greedy_exact = []
        self._greedy_pos_ratio = []
        self._greedy_pos_diff = []
        self._greedy_tok_acc = []   # optional

    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero:
            return
        if len(self._greedy_eos_rate) == 0:
            return

        mean = lambda xs: torch.stack(xs).mean()

        self.log("val/greedy_eos_rate", mean(self._greedy_eos_rate), prog_bar=True)
        self.log("val/greedy_no_eos_rate", mean(self._greedy_no_eos_rate), prog_bar=True)
        self.log("val/greedy_len_ratio", mean(self._greedy_len_ratio), prog_bar=True)
        self.log("val/greedy_eos_pos_diff", mean(self._greedy_pos_diff))
        self.log("val/greedy_eos_pos_ratio", mean(self._greedy_pos_ratio))
        self.log("val/greedy_exact_match", mean(self._greedy_exact))
        self.log("val/greedy_token_acc", mean(self._greedy_tok_acc))

    @staticmethod
    def _first_eos_pos(seq: torch.Tensor, eos_id: int):
        """
        seq: [B,T]
        returns:
          pos: [B] first eos index (0..T-1) or T if none
          has: [B] bool
        """
        T = seq.size(1)
        has = (seq == eos_id).any(dim=1)
        pos = (seq == eos_id).float().argmax(dim=1)  # safe because we also use has
        pos = torch.where(has, pos, torch.full_like(pos, T))
        return pos, has

    @staticmethod
    def _eos_aware_len(seq: torch.Tensor, eos_id: int, pad_id: int):
        """
        length = eos_pos+1 if has_eos else count(non-pad)
        """
        eos_pos, has = LitHSQC2Selfies._first_eos_pos(seq, eos_id)
        nonpad_len = (seq != pad_id).sum(dim=1)
        length = torch.where(has, eos_pos + 1, nonpad_len)
        return length, has, eos_pos

    @staticmethod
    def _exact_match_eos_aware(gen: torch.Tensor, tgt: torch.Tensor, eos_id: int, pad_id: int):
        """
        strict: sequences must match token-by-token up to eos (inclusive) and lengths must match.
        Only meaningful when both have eos.
        """
        gen_len, gen_has, gen_eos_pos = LitHSQC2Selfies._eos_aware_len(gen, eos_id, pad_id)
        tgt_len, tgt_has, tgt_eos_pos = LitHSQC2Selfies._eos_aware_len(tgt, eos_id, pad_id)

        mask = gen_has & tgt_has
        if not mask.any():
            return torch.tensor(float("nan"), device=gen.device), mask, gen_len, tgt_len, gen_eos_pos, tgt_eos_pos

        # compare up to maxT = min(Tgen, Ttgt) but we also enforce same_len
        maxT = min(gen.size(1), tgt.size(1))
        equal_prefix = (gen[:, :maxT] == tgt[:, :maxT]).all(dim=1)

        same_len = (gen_len == tgt_len)
        exact = (equal_prefix & same_len & mask).float()

        # mean over masked items only
        exact_mean = exact[mask].mean()
        return exact_mean, mask, gen_len, tgt_len, gen_eos_pos, tgt_eos_pos

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
