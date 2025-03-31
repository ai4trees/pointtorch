"""Utilities for configuring the package setup."""

__all__ = ["open3d_is_available", "pytorch3d_is_available", "flash_attention_is_available"]


def flash_attention_is_available():
    """
    Returns:
        :code:`True` if FlashAttention is available and :code:`False` otherwise.
    """
    flash_attention_available = False
    try:
        import flash_attn as _  # type: ignore #pylint: disable=import-outside-toplevel

        flash_attention_available = True
    except (ModuleNotFoundError, TypeError):
        pass

    return flash_attention_available


def open3d_is_available():
    """
    Returns: :code:`True` if Open3D is installed and :code:`False` otherwise.
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
    Returns: :code:`True` if PyTorch3D is installed and :code:`False` otherwise.
    """

    pytorch3d_available = False
    try:
        import pytorch3d as _  # pylint: disable=import-outside-toplevel

        pytorch3d_available = True
    except (ModuleNotFoundError, TypeError):
        pass

    return pytorch3d_available
