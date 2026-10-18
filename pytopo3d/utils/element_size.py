"""Physical dimensions for uniform grids of rectangular H8 elements.

Element dimensions always use global XYZ order, independently of the YXZ
storage order of density arrays. All lengths are in meters.
"""

from typing import Optional, Sequence, Union

import numpy as np


ElementSize = Union[float, Sequence[float]]
DEFAULT_ELEMENT_SIZE = 0.01


def element_size_xyz(elem_size: ElementSize) -> tuple[float, float, float]:
    """Validate a scalar or three XYZ edge lengths and return an XYZ tuple."""
    sizes = np.asarray(elem_size, dtype=float)
    if sizes.ndim == 0:
        sizes = np.repeat(sizes, 3)
    if sizes.shape != (3,) or not np.all(np.isfinite(sizes)) or np.any(sizes <= 0):
        raise ValueError(
            "elem_size must be a finite positive scalar or three finite positive "
            f"XYZ edge lengths, got {elem_size}"
        )
    return tuple(float(value) for value in sizes)


def resolve_element_size(
    nelx: int,
    nely: int,
    nelz: int,
    elem_size: Optional[ElementSize] = None,
    *,
    x_axis_scale: Optional[float] = None,
    y_axis_scale: Optional[float] = None,
    z_axis_scale: Optional[float] = None,
    target_physical_x: Optional[float] = None,
) -> ElementSize:
    """Resolve domain lengths using the final (possibly imported) grid counts.

    An omitted axis retains the legacy 0.01 m element edge length. Scalar sizing
    and legacy target-physical-x sizing continue to return a scalar.
    """
    counts = (nelx, nely, nelz)
    for name, count in zip(("nelx", "nely", "nelz"), counts):
        if not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError(f"{name} must be a positive integer, got {count}")
    scales = (x_axis_scale, y_axis_scale, z_axis_scale)
    axis_sizing = any(value is not None for value in scales)
    if axis_sizing and elem_size is not None:
        raise ValueError("--elem-size cannot be combined with any --x/y/z-axis-scale flag")
    if axis_sizing and target_physical_x is not None:
        raise ValueError(
            "--target-physical-x cannot be combined with any --x/y/z-axis-scale flag"
        )
    for name, value in zip(
        ("--x-axis-scale", "--y-axis-scale", "--z-axis-scale", "--target-physical-x"),
        (*scales, target_physical_x),
    ):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive, got {value}")
    if elem_size is not None:
        element_size_xyz(elem_size)
    if axis_sizing:
        return element_size_xyz(tuple(
            DEFAULT_ELEMENT_SIZE if length is None else length / count
            for length, count in zip(scales, counts)
        ))
    if target_physical_x is not None:
        size = target_physical_x / nelx
        element_size_xyz(size)
        return size
    return DEFAULT_ELEMENT_SIZE if elem_size is None else elem_size


def resolve_cli_element_size(args) -> ElementSize:
    """Resolve parsed CLI sizing after geometry loading has set the grid counts.

    The parser already rejects explicit conflicting flags. Its default elem_size
    is ignored only when an axis length was requested.
    """
    scales = {
        f"{axis}_axis_scale": getattr(args, f"{axis}_axis_scale", None)
        for axis in "xyz"
    }
    size = None if any(value is not None for value in scales.values()) else args.elem_size
    return resolve_element_size(
        args.nelx, args.nely, args.nelz, size,
        target_physical_x=getattr(args, "target_physical_x", None), **scales,
    )


def geometry_metadata(
    nelx: int,
    nely: int,
    nelz: int,
    elem_size: ElementSize,
    *,
    x_axis_scale: Optional[float] = None,
    y_axis_scale: Optional[float] = None,
    z_axis_scale: Optional[float] = None,
) -> dict:
    """JSON-compatible requested lengths and resolved physical geometry."""
    sizes = element_size_xyz(elem_size)
    return {
        "x_axis_scale": x_axis_scale,
        "y_axis_scale": y_axis_scale,
        "z_axis_scale": z_axis_scale,
        "element_size_xyz_m": list(sizes),
        "domain_size_xyz_m": [count * size for count, size in zip((nelx, nely, nelz), sizes)],
        "element_volume_m3": float(np.prod(sizes)),
    }
