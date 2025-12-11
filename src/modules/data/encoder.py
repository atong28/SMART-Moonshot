from typing import List, Optional, Sequence, Tuple

import torch
import numpy as np

from ..log import get_logger


class FourierFeatures(torch.nn.Module):
    """
    Scalar -> sinusoidal embedding:

        x -> [sin(x / s_0), ..., sin(x / s_{n_sin-1}),
              cos(x / c_0), ..., cos(x / c_{n_cos-1})]

    Parameters
    ----------
    dim_model : int
        Number of output features.
    min_wavelength : float
        Minimum wavelength in the log-spaced range.
    max_wavelength : float
        Maximum wavelength in the log-spaced range.
    """

    def __init__(
        self,
        dim_model: int,
        min_wavelength: float = 1e-3,
        max_wavelength: float = 1e4,
    ):
        super().__init__()

        n_sin = dim_model // 2
        n_cos = dim_model - n_sin

        if min_wavelength:
            base = min_wavelength / (2 * np.pi)
            scale = max_wavelength / min_wavelength
        else:
            base = 1.0
            scale = max_wavelength / (2 * np.pi)

        sin_den = max(n_sin - 1, 1)
        cos_den = max(n_cos - 1, 1)

        sin_term = base * scale ** (
            torch.arange(0, n_sin).float() / sin_den
        )
        cos_term = base * scale ** (
            torch.arange(0, n_cos).float() / cos_den
        )

        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (..., 1)
            Scalar coordinates to encode.

        Returns
        -------
        torch.Tensor of shape (..., dim_model)
            Sinusoidal embeddings.
        """
        if self.sin_term.device != x.device or self.sin_term.dtype != x.dtype:
            self.sin_term = self.sin_term.to(device=x.device, dtype=x.dtype)
            self.cos_term = self.cos_term.to(device=x.device, dtype=x.dtype)

        sin_mz = torch.sin(x / self.sin_term)
        cos_mz = torch.cos(x / self.cos_term)
        return torch.cat([sin_mz, cos_mz], dim=-1)


class CoordinateEncoder(torch.nn.Module):
    """
    Positional encoding for coordinates, with optional sign-encoding dimensions.

    This merges the old CoordinateEncoder + SignCoordinateEncoder into a single
    generic encoder.

    Parameters
    ----------
    dim_model : int
        The total output dimensionality.
    dim_coords : Sequence[int]
        Per-input-dimension embedding sizes (for bookkeeping). The entries
        corresponding to sign dimensions are *not* used directly; instead the
        sign embedding size is computed as:

            sign_embedding_size = dim_model - sum(dim for non-sign dims)

        So:
            sum(non-sign dims) + sign_embedding_size == dim_model
    wavelength_bounds : list[tuple[float, float]] or None
        Bounds (min, max) for wavelengths for each *non-sign* dimension,
        in order of appearance. This matches the behavior of the original
        SignCoordinateEncoder: the list is consumed only for non-sign dims.
    is_sign_encoding : list[bool] or None
        Flags indicating which coordinates use sign encoding. At most one
        dimension may be sign-encoded.
    """

    def __init__(
        self,
        dim_model: int,
        dim_coords: Sequence[int],
        wavelength_bounds: Optional[Sequence[Tuple[float, float]]] = None,
        is_sign_encoding: Optional[Sequence[bool]] = None,
    ):
        super().__init__()
        self.logger = get_logger(__file__)

        dim_coords = list(dim_coords)
        self.dim_coords = dim_coords

        if is_sign_encoding is None:
            is_sign_encoding = [False] * len(dim_coords)
        else:
            is_sign_encoding = list(is_sign_encoding)

        assert len(is_sign_encoding) == len(dim_coords), (
            f"is_sign_encoding length ({len(is_sign_encoding)}) must match "
            f"dim_coords length ({len(dim_coords)})"
        )

        num_sign_dims = sum(is_sign_encoding)
        assert num_sign_dims <= 1, (
            "CoordinateEncoder supports at most one sign encoding dimension, "
            f"but found {num_sign_dims}"
        )

        self.is_sign_encoding = is_sign_encoding

        self.sign_dim_idx: Optional[int] = None
        non_sign_dims_sum = 0
        for idx, (dim, is_sign) in enumerate(zip(dim_coords, is_sign_encoding)):
            if is_sign:
                self.sign_dim_idx = idx
            else:
                non_sign_dims_sum += dim

        if self.sign_dim_idx is not None:
            self.sign_embedding_size = dim_model - non_sign_dims_sum
            assert self.sign_embedding_size >= 0, (
                "Sign embedding size became negative. "
                "Check dim_model vs non-sign dims."
            )
        else:
            self.sign_embedding_size = 0
            assert non_sign_dims_sum == dim_model, (
                "With no sign dimension, sum(dim_coords) must equal dim_model."
            )

        self.dim_to_wavelength = {}
        if wavelength_bounds is not None:
            wavelength_idx = 0
            for idx, (dim, is_sign) in enumerate(zip(dim_coords, is_sign_encoding)):
                if not is_sign and dim > 0:
                    assert wavelength_idx < len(wavelength_bounds), (
                        "Not enough wavelength_bounds provided for non-sign dims."
                    )
                    self.dim_to_wavelength[idx] = wavelength_bounds[wavelength_idx]
                    wavelength_idx += 1

            assert wavelength_idx == len(wavelength_bounds), (
                "Too many wavelength_bounds provided for the number of non-sign dims."
            )

        encoders = []
        for idx, (dim, is_sign) in enumerate(zip(dim_coords, is_sign_encoding)):
            if is_sign or dim == 0:
                encoders.append(None)
                continue

            if idx in self.dim_to_wavelength:
                min_w, max_w = self.dim_to_wavelength[idx]
                enc = FourierFeatures(dim_model=dim, min_wavelength=min_w, max_wavelength=max_w)
            else:
                enc = FourierFeatures(dim_model=dim)
            encoders.append(enc)

        self.positional_encoders = torch.nn.ModuleList(
            [e if e is not None else torch.nn.Identity() for e in encoders]
        )
        self._encoder_is_none = [e is None for e in encoders]

        self.logger.debug(
            f"Initialized CoordinateEncoder[{dim_model}] with dims {self.dim_coords}, "
            f"is_sign_encoding={self.is_sign_encoding}, "
            f"{sum(not f for f in self._encoder_is_none)} positional encoders, "
            f"and {self.sign_embedding_size} total bits for sign encoding."
        )

        self.dim_model = dim_model

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : Tensor of shape (B*N, D)
            Coordinates to encode, where D must equal len(dim_coords).

        Returns
        -------
        Tensor of shape (B*N, dim_model)
        """
        assert X.shape[1] == len(self.dim_coords), (
            f"Input dimension {X.shape[1]} must match dim_coords length "
            f"{len(self.dim_coords)}"
        )

        embeddings = []

        for idx, (is_sign, encoder_is_none, encoder) in enumerate(
            zip(self.is_sign_encoding, self._encoder_is_none, self.positional_encoders)
        ):
            if is_sign:
                sign_dim = X[:, idx:idx + 1]  # (B*N, 1)
                encoded = torch.sign(sign_dim).expand(-1, self.sign_embedding_size)
                embeddings.append(encoded)
            elif not encoder_is_none:
                coord = X[:, idx:idx + 1]  # (B*N, 1)
                embeddings.append(encoder(coord))  # (B*N, dim_for_dim)
            else:
                continue

        out = torch.cat(embeddings, dim=1)
        assert out.shape[1] == self.dim_model, (
            f"Output dimension {out.shape[1]} does not match dim_model {self.dim_model}"
        )
        return out


def build_encoder(
    dim_model: int,
    dim_coords: List[int],
    wavelength_bounds: Optional[Sequence[Tuple[float, float]]] = None,
    is_sign_encoding: Optional[Sequence[bool]] = None,
) -> CoordinateEncoder:
    """
    Convenience factory, matching the old build_encoder signature.
    """
    return CoordinateEncoder(
        dim_model=dim_model,
        dim_coords=dim_coords,
        wavelength_bounds=wavelength_bounds,
        is_sign_encoding=is_sign_encoding,
    )
