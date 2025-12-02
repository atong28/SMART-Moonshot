import logging
from typing import List
import torch
import numpy as np

class PositionalEncoder(torch.nn.Module):
    """Encode positions using sine and cosine waves.
    Parameters
    ----------
    dim_model : int
        The number of features to output.
    min_wavelength : float
        The minimum wavelength to use. 
    max_wavelength : float
        The maximum wavelength to use.
    """

    def __init__(
        self,
        dim_model=128,
        min_wavelength=.001,
        max_wavelength=10000
    ):
        super().__init__()

        n_sin = int(dim_model / 2)
        n_cos = dim_model - n_sin

        if min_wavelength:
            base = min_wavelength / (2 * np.pi)
            scale = max_wavelength / min_wavelength
        else:
            base = 1
            scale = max_wavelength / (2 * np.pi)

        sin_term = base * scale ** (
            torch.arange(0, n_sin).float() / (n_sin - 1)
        )
        cos_term = base * scale ** (
            torch.arange(0, n_cos).float() / (n_cos - 1)
        )

        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, X):
        """Encode positions
        Parameters
        ----------
        X : torch.Tensor of shape (n_positions)
            The positions to encode
        Returns
        -------
        torch.Tensor of shape (n_positions, dim_model)
            The encoded positions
        """
        if torch.cuda.is_available():
            self.sin_term = self.sin_term.to(X)
            self.cos_term = self.cos_term.to(X)
        
        sin_mz = torch.sin(X / self.sin_term)
        cos_mz = torch.cos(X / self.cos_term)
        return torch.cat([sin_mz, cos_mz], axis=-1)

class CoordinateEncoder(torch.nn.Module):
    """
    Generate positional encoding of coordinates

    Parameters
    ----------
    dim_model : int, optional
        The latent dimensionality used by the Transformer model.
    dim_coords: tuple or None
        A tuple specifying the number of features to use for encoding 
        each coordinate. Must sum to dim model
    wavelength_bounds : list(tuple), optional
        A list of tuples of (minimum, maximum) wavelengths for
        each dimension to be encoded 
    """

    def __init__(
        self,
        dim_model,
        dim_coords,
        wavelength_bounds=None,
    ):
        super().__init__()
        assert (sum(dim_coords) == dim_model)
        if wavelength_bounds:
            assert (len(wavelength_bounds) == len(dim_coords))

        self.positional_encoders = []
        self.dim_coords = dim_coords
        for idx, dim in enumerate(dim_coords):
            if wavelength_bounds:
                min_wavelength = wavelength_bounds[idx][0]
                max_wavelength = wavelength_bounds[idx][1]
                p = PositionalEncoder(
                    dim_model=dim,
                    min_wavelength=min_wavelength,
                    max_wavelength=max_wavelength
                )
            else:
                p = PositionalEncoder(dim_model=dim)
            self.positional_encoders.append(p)

    def forward(self, X):
        """Encode coordinates
        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_coords, n_dimensions)
            The coordinates to embed
        Returns
        -------
        torch.Tensor of shape (batch_size, n_coords, dim_model)
            The encoded coordinates
        """
        assert (X.shape[2] == len(self.dim_coords))
        embeddings = []
        for dim, encoder in enumerate(self.positional_encoders):
            embeddings.append(encoder(X[:, :, [dim]]))

        return torch.cat(embeddings, dim=2)



class SignCoordinateEncoder(torch.nn.Module):
    """
    Positional encoding for coordinates, with optional sign encoding dimensions.

    Expects input X of shape (B*N, D), where D can vary based on input data.
    Uses is_sign_encoding to specify which dimensions use sign encoding.
    """
    def __init__(
        self,
        dim_model,
        dim_coords,
        wavelength_bounds=None,
        is_sign_encoding=None
    ):
        super().__init__()
        self.logger = logging.getLogger("lightning")
        if is_sign_encoding is None:
            is_sign_encoding = [False] * len(dim_coords)
        
        assert len(is_sign_encoding) == len(dim_coords), \
            f"is_sign_encoding length ({len(is_sign_encoding)}) must match dim_coords length ({len(dim_coords)})"
        
        # Assert that there is at most one sign encoding dimension
        num_sign_dims = sum(is_sign_encoding)
        assert num_sign_dims <= 1, \
            f"SignCoordinateEncoder supports at most one sign encoding dimension, but found {num_sign_dims}"
        
        self.is_sign_encoding = is_sign_encoding
        self.dim_coords = dim_coords
        
        # Find sign dimension and calculate sizes in one pass
        self.sign_dim_idx = None
        non_sign_dims_sum = 0
        for idx, (dim, is_sign) in enumerate(zip(dim_coords, is_sign_encoding)):
            if is_sign:
                self.sign_dim_idx = idx
            else:
                non_sign_dims_sum += dim
        
        self.sign_embedding_size = dim_model - non_sign_dims_sum if self.sign_dim_idx is not None else 0
        
        # Build mapping of dimension index to wavelength bounds for non-sign dimensions
        self.dim_to_wavelength = {}
        if wavelength_bounds:
            wavelength_idx = 0
            for idx, (dim, is_sign) in enumerate(zip(dim_coords, is_sign_encoding)):
                if not is_sign and dim > 0:
                    self.dim_to_wavelength[idx] = wavelength_bounds[wavelength_idx]
                    wavelength_idx += 1
        
        # Build positional encoders
        self.positional_encoders = []
        for idx, (dim, is_sign) in enumerate(zip(dim_coords, is_sign_encoding)):
            if is_sign or dim == 0:
                # Sign encoding or zero dimension - no positional encoder
                self.positional_encoders.append(None)
            else:
                # Regular positional encoding dimension
                if idx in self.dim_to_wavelength:
                    min_wavelength, max_wavelength = self.dim_to_wavelength[idx]
                    p = PositionalEncoder(
                        dim_model=dim,
                        min_wavelength=min_wavelength,
                        max_wavelength=max_wavelength
                    )
                else:
                    p = PositionalEncoder(dim_model=dim)
                self.positional_encoders.append(p)

        self.logger.debug(
            f"Initialized SignCoordinateEncoder[{dim_model}] with dims {self.dim_coords}, "
            f"is_sign_encoding={self.is_sign_encoding}, "
            f"{len([p for p in self.positional_encoders if p is not None])} positional encoders, "
            f"and {self.sign_embedding_size} total bits for sign encoding."
        )

    def forward(self, X):
        """
        Parameters
        ----------
        X : Tensor of shape (B*N, D)
            Coordinates to encode, where D must equal len(dim_coords)
        
        Returns
        -------
        Tensor of shape (B*N, dim_model)
        """
        assert X.shape[1] == len(self.dim_coords), \
            f"Input dimension {X.shape[1]} must match dim_coords length {len(self.dim_coords)}"
        
        embeddings = []
        
        for idx, (encoder, is_sign) in enumerate(zip(self.positional_encoders, self.is_sign_encoding)):
            if is_sign:
                # Sign encoding dimension
                sign_dim = X[:, idx:idx+1]  # shape (B*N, 1)
                encoded = torch.sign(sign_dim).expand(-1, self.sign_embedding_size)
                embeddings.append(encoded)
            elif encoder is not None:
                # Positional encoding dimension
                coord = X[:, idx:idx+1]  # shape (B*N, 1)
                embeddings.append(encoder(coord))  # shape (B*N, dim_for_dim)
        
        return torch.cat(embeddings, dim=1)

def build_encoder(
    dim_model: int, 
    dim_coords: List[int],
    wavelength_bounds=None,
    is_sign_encoding=None
):
    return SignCoordinateEncoder(dim_model, dim_coords, wavelength_bounds, is_sign_encoding)