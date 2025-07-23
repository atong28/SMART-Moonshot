import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from .src.settings import Args
from .src.dataset import MoonshotDataModule
from .src.model import SPECTRE, OptionalInputSPECTRE


def visualize(data_module: MoonshotDataModule, model: SPECTRE | OptionalInputSPECTRE, ckpt_path: str = None):
    assert ckpt_path, 'Cannot visualize without pretrained model'

    # Load test data
    data_module.setup('test')
    test_loader = data_module.test_dataloader()

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize accumulation
    modality_attn_total = defaultdict(float)      # total attention weight per modality
    modality_token_counts = defaultdict(int)      # total number of tokens per modality
    type_names = ["HSQC", "C_NMR", "H_NMR", "MW", "ISO", "MS"]

    with torch.no_grad():
        for batch in test_loader:
            inputs, _, type_indicator = batch
            inputs = inputs.to(device)
            type_indicator = type_indicator.to(device)

            attn_weights, pad_mask, type_indicator_cpu = model.forward_with_attention(inputs, type_indicator)

            # First layer attention: (B, num_heads, 1, N)
            layer_attn = attn_weights[0]
            B = layer_attn.size(0)

            for b in range(B):
                attn = layer_attn[b].mean(0).squeeze(0).cpu().numpy()  # (N,)
                types = type_indicator_cpu[b].numpy()
                mask = pad_mask[b].numpy()

                for i, (t, m) in enumerate(zip(types, mask)):
                    if not m:
                        modality_attn_total[int(t)] += attn[i]
                        modality_token_counts[int(t)] += 1

    # Normalize attention per modality by total attention sum
    total_attn_sum = sum(modality_attn_total.values())
    attn_fraction = [
        modality_attn_total[t] / total_attn_sum if total_attn_sum > 0 else 0.0
        for t in range(len(type_names))
    ]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(type_names, attn_fraction)
    plt.ylabel("Fraction of Total Attention")
    plt.title("CLS Attention to Modalities (Summed Over Tokens and Test Set)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualization.png')
    print("[✓] Saved attention visualization to 'visualization.png'")
