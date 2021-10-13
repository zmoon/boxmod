"""
Loaders for included mechanisms by name.
"""
from pathlib import Path

THIS_DIR = Path(__file__).parent

# Mechs in my YAML format
_YML_MECHS = {
    "Stockwell and Goliff (2002)": "Stockwell-and-Goliff-2002.yml",
}

_ALL_MECH_NAMES = list(_YML_MECHS.keys())  # + ...


def load_mech(name: str):
    """Load mechanism equations and return an `EqnSet`.

    Parameters
    ----------
    name : str
        Name of the mechanism.
    """
    from .. import load
    from ..eqns import EqnSet

    if name in _YML_MECHS:
        eqns = load.read_yaml(THIS_DIR / _YML_MECHS[name])

    else:
        raise ValueError(
            f"invalid mech name {name!r}. "
            f"Valid options are: {', '.join(f'{n!r}' for n in _ALL_MECH_NAMES)}"
        )

    return EqnSet(eqns, long_name=name)
