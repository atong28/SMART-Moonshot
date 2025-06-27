import logging
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

        self.sin_terms = sin_term
        self.cos_terms = cos_term

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
            print(wavelength_bounds)
            print(dim_coords)
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
                print(f"Pushed an encoder with bounds {min_wavelength}, {max_wavelength}")
            else:
                p = PositionalEncoder(dim_model=dim)
                print(f"Pushed an encoder with no defined bounds")
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
    Generate positional encoding of coordinates, except the final bit
    is binary encoded as 0 or 1. 

    Thus, dim_coords and wavelength_bounds should only encode for 
    n-1 dimensions, and the amount of bits for the final dimension can be interpolated

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
        use_peak_values=False
    ):
        super().__init__()
        self.logger = logging.getLogger("lightning")
        assert (sum(dim_coords) == dim_model)
        if wavelength_bounds:
            assert (len(wavelength_bounds) == len(dim_coords) - 1)
        self.sign_embedding_dims = dim_model - sum(dim_coords[:-1])
        self.positional_encoders = []
        self.dim_coords = dim_coords
        self.use_peak_values = use_peak_values
        for idx, dim in enumerate(dim_coords[:-1]):
            if wavelength_bounds:
                min_wavelength = wavelength_bounds[idx][0]
                max_wavelength = wavelength_bounds[idx][1]
                p = PositionalEncoder(dim_model=dim,
                                    min_wavelength=min_wavelength,
                                    max_wavelength=max_wavelength)
                # self.logger.warning(
                #     f"Pushed an encoder with bounds {min_wavelength}, {max_wavelength}")
                # self.logger.warning(
                #     f"Sin wavelengths: {str(np.round(2 * torch.pi / p.sin_terms, 3))}")
                # self.logger.warning(
                #     f"Cos wavelengths: {str(np.round(2 * torch.pi / p.cos_terms, 3))}")
            else:
                p = PositionalEncoder(dim_model=dim)
                self.logger.warning(
                    f"Pushed an encoder with no bounds")
            self.positional_encoders.append(p)
        self.logger.warning(f"Initialized SignCoordinateEncoder[{dim_model}] with dims " +
                            f"{self.dim_coords} and {len(self.positional_encoders)} positional encoders. " +
                            f"{self.sign_embedding_dims} bits are reserved for encoding the final bit")

    def forward(self, X):
        """Encode coordinates
        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_coords, n_dimensions) (32*50*3)
            The coordinates to embed
        Returns
        -------
        torch.Tensor of shape (batch_size, n_coords, dim_model)
            The encoded coordinates
        """
        assert (X.shape[2] == len(self.dim_coords))
        embeddings = []
        for dim, encoder in enumerate(self.positional_encoders):
            embeddings.append(encoder(X[:, :, [dim]])) # 32*50*180, 32*50*180, 32*50*24
        if self.sign_embedding_dims:
            if self.use_peak_values:
                peaks = X[:, :, [-1] * self.sign_embedding_dims].float() #   
                embeddings.append(peaks)
                # print(peaks)
                # exit(0)
            else:
                # signs = torch.where(X[:, :, [-1] * self.sign_embedding_dims] >= 0, 1, -1).float()
                multiplicity_values = X[:, :, [-1] * self.sign_embedding_dims]
                signs = torch.sign(multiplicity_values).float()
                embeddings.append(signs)
                # print(signs)
                # exit(0)
        return torch.cat(embeddings, dim=2)

def build_encoder(
    dim_model: int, 
    dim_coords: tuple[int, int, int],
    wavelength_bounds=None,
    use_peak_values=False
):
    return SignCoordinateEncoder(dim_model, dim_coords, wavelength_bounds, use_peak_values)