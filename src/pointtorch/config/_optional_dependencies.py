""" Utilities for configuring the package setup. """

__all__ = ["open3d_is_available", "pytorch3d_is_available"]


def open3d_is_available():
    """
    Returns: `True` if Open3D is installed and `False` otherwise.
    """

    open3d_available = False
    try:
        from open3d.ml import torch as _  # pylint: disable=import-outside-toplevel

        open3d_available = True
    except (ModuleNotFoundError, TypeError, Exception):  # pylint: disable=broad-exception-caught
        pass

    return open3d_available


def pytorch3d_is_available() -> bool:
    """
    Returns: `True` if PyTorch3D is installed and `False` otherwise.
    """

    pytorch3d_available = False
    try:
        import pytorch3d as _  # pylint: disable=import-outside-toplevel

        pytorch3d_available = True
    except (ModuleNotFoundError, TypeError):
        pass

    return pytorch3d_available
