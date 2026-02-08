
import torch

def collate_hsqc_selfies(batch, pad_id=0):
    # batch: list[(hsqc[N,3], ids[T])]
    B = len(batch)
    n_max = max(x[0].shape[0] for x in batch)
    t_max = max(x[1].shape[0] for x in batch)

    peaks = torch.zeros(B, n_max, 3, dtype=torch.float32)
    peaks_mask = torch.zeros(B, n_max, dtype=torch.bool)

    ids_pad = torch.full((B, t_max), pad_id, dtype=torch.long)
    ids_mask = torch.zeros(B, t_max, dtype=torch.bool)

    for i, (hsqc, ids) in enumerate(batch):
        n = hsqc.shape[0]
        t = ids.shape[0]

        peaks[i, :n] = hsqc
        peaks_mask[i, :n] = True

        ids_pad[i, :t] = ids
        ids_mask[i, :t] = True

    # teacher forcing shift:
    # y_in  = [BOS, ..., token_{T-2}]
    # y_out = [token_1, ..., EOS]
    y_in  = ids_pad[:, :-1]
    y_out = ids_pad[:, 1:]
    y_mask = ids_mask[:, 1:]  # valid target positions

    return peaks, peaks_mask, y_in, y_out, y_mask
