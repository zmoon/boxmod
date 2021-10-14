"""
Load and parse equations from files specifying them.
The ``read_*`` functions return lists of `Eqn` objects,
which can then be used to create `EqnSet`s.
"""
import warnings
from typing import List
from typing import NamedTuple
from typing import Optional

import numpy as np

from .eqns import Eqn

__all__ = ("read_csvs", "read_yaml", "read_yaml_permm")


# TODO: Eqn.from_string static method, where separator (or separator pattern) could be passed


class _MechData(NamedTuple):
    """Data needed to init an `EqnSet`."""

    equations: List[Eqn]
    name: Optional[str] = None
    long_name: Optional[str] = None
    species: Optional[List[str]] = None


def parse_eqn_side(s):
    """Split on `+` and extract species and stoich. coeffs.

    s : str
        of a form like `A + B` or `2 A + B` or `0.35 C + 0.5 D + 1.1 E`

    Returns
    -------
    list
        of (mult, spc) tuples.
        Can be used to initialize `EqnElement`s
        directly or through `Eqn`.
    """
    # split on `+`
    side_elements = [_.strip() for _ in s.split("+")]

    ret = []  # collect (mult, spc) tuples here
    for side_element in side_elements:
        if " " in side_element:  # multiplier, i.e., stoich coeff or yield. must have the space!
            mult, spc = side_element.split(" ")
            mult = float(mult)
        # TODO: allow space to be one or more?
        # TODO: if mult is negative, raise error or correct and warn? (KPP allows -1 but it has a certain meaning)

        else:  # no multiplier
            mult = 1.0
            spc = side_element

        # number at beginning of species probably indicates error in mech
        if spc[0].isnumeric():
            msg = (
                f"Species detected that starts with number: {spc}"
                "\nIf this was supposed to be a stoich coeff, add a space."
            )
            warnings.warn(msg)

        # TODO: maybe such filtering should be in Eqn or EqnSet?
        if spc in ["hv", "hν"]:  # second one is actual nu, not v
            print(f"Note hν detected and skipped in {' + '.join(side_elements)}")
            continue

        ret.append((mult, spc))

    return ret


def _read_csv(fp):
    """Read an equations CSV file.
    The first 3 columns should be: reactants, products, k.
    The reactants and products should be joined by `` + ``.

    Parameters
    ----------
    fp
        Path-like to the CSV file.

    Returns
    -------
    list(Eqn)
    """
    # TODO: could just use builtin csv module
    eqn_arr = np.loadtxt(fp, delimiter=",", usecols=(0, 1, 2), dtype=str)  # , encoding='utf-8-sig')

    rct_arr = eqn_arr[:, 0]
    pdt_arr = eqn_arr[:, 1]
    k_arr = eqn_arr[:, 2].astype(np.float)

    eqns = []
    for rcts, pdts, k in zip(rct_arr, pdt_arr, k_arr):
        ms_rcts = parse_eqn_side(rcts)
        ms_pdts = parse_eqn_side(pdts)

        eqns.append(Eqn(ms_rcts, ms_pdts, k))

    return eqns


def read_csvs(fp_list):
    """Read CSVs from a list of filepaths and combine the lists of reactions.
    The first 3 columns should be: reactants, products, k.
    The reactants and products should be joined by `` + ``.

    Parameters
    ----------
    fp_list : list
        List of path-likes corresponding to the CSV files.

    Returns
    -------
    list(Eqn)
    """
    if not isinstance(fp_list, list):
        fp_list = [fp_list]

    # Add all
    eqns0 = []
    for fp in fp_list:
        eqns0.extend(_read_csv(fp))

    # Check for dupes
    eqns = []
    for eqn in eqns0:
        if eqn in eqns:
            print(f"{eqn} already in the list. Skipping.")
        else:
            eqns.append(eqn)

    return eqns


# Construct exp photolysis fns
def _gen_simple_exp_photol_fns():
    """a exp(-0.575/cos(SZA))"""
    import math  # noqa: F401

    data = {
        "o3_o1d": 3.83e-5,
        "no2": 1.67e-2,
        "hcho_h": 3.55e-5,  # HCHO -> 2 HO2 + CO
        "hcho_h2": 4.91e-5,  # TODO: fix with correct value
        "mrooh": 3.01e-5,
    }

    fns = {}
    for id_, a in data.items():
        s_f = f"""
def f(SZA):
    if SZA > math.pi / 2:  # > 90 deg.
        return 0
    else:
        return {a} * math.exp(-0.575 / math.cos(SZA))
        """.strip()

        exec(s_f, locals())

        fns[id_] = locals()["f"]

    return fns


_EXP_PHOTOL_FNS = _gen_simple_exp_photol_fns()


def _gen_k_fn_yml(s: str):
    import math  # noqa: F401

    # Replace certain things so valid Python
    rpls = {
        "[M]": "M",
        "^": "**",
        "exp": "math.exp",
    }
    s_f_body = s
    for s1, s2 in rpls.items():
        s_f_body = s_f_body.replace(s1, s2)

    # Get args
    args0 = ["T", "M"]  # desired order
    args = [arg for arg in args0 if arg in s_f_body]

    s_f = f"""
def f({', '.join(args)}):
    return {s_f_body}
    """.strip()

    exec(s_f, locals())

    return locals()["f"]


def _gen_k_fn_yml_multiline(s: str):
    import math  # noqa: F401

    rpls = {
        "[M]": "M",
        "^": "**",
        "exp": "math.exp",
        "log10": "math.log10",
    }
    s_ = s
    for s1, s2 in rpls.items():
        s_ = s_.replace(s1, s2)

    lines = s_.splitlines()

    assert lines[-1].startswith("k = ")
    ret = lines[-1].partition(" = ")[-1]

    s_f_body = "\n".join(lines[:-1]) + f"\nreturn {ret}"
    s_f_body = "\n".join(f"    {line}" for line in s_f_body.splitlines())  # indent

    # Get args
    args0 = ["T", "M"]  # desired order
    args = [arg for arg in args0 if arg in s_f_body]

    s_f = f"""
def f({', '.join(args)}):
{s_f_body}
    """.strip()

    exec(s_f, locals())

    return locals()["f"]


def _gen_k_fn_float(x: float):
    s_f = f"""
def f():
    return {x}
    """.strip()

    exec(s_f, locals())

    return locals()["f"]


def read_yaml(fp) -> _MechData:
    """Load a mech in my YAML format with equation and rate exprs separate.

    Parameters
    ----------
    fp
        Path-like to the YAML file.
    """
    import yaml

    with open(fp, "r") as f:
        data = yaml.load(f, Loader=yaml.Loader)

    eqns = []
    for d in data["reactions"]:

        lhs, rhs = d["eqn"].split("->")

        # Reactants
        ms_rcts = parse_eqn_side(lhs)  # reactant (mult, spc)

        # Products
        ms_pdts = parse_eqn_side(rhs)  # product (mult, spc)

        # Rate coeff
        k0 = d.get("k0", d["k"])
        assert isinstance(k0, float)

        # Rate expr
        k_expr0 = d["k"]

        # Single number
        if isinstance(k_expr0, float):
            f = _gen_k_fn_float(k_expr0)

        # Reference to photol reaction
        elif k_expr0.startswith("j_"):
            f = _EXP_PHOTOL_FNS[k_expr0[2:]]

        # Reference to longer expression defined in the mech
        elif k_expr0.startswith("k"):
            f = _gen_k_fn_yml_multiline(data["k_expressions"][k_expr0])

        # Single-line expression
        else:
            f = _gen_k_fn_yml(k_expr0)

        # Form equation
        eqn = Eqn(ms_rcts, ms_pdts, k0, k_fn=f)

        eqns.append(eqn)

    name = data.get("name")
    long_name = data.get("long_name")

    return _MechData(equations=eqns, name=name, long_name=long_name)


def read_yaml_permm(fp):
    """Read a permm mechanism (no rate constants but some additional info).

    Parameters
    ----------
    fp
        Path-like to the YAML file.

    Returns
    -------
    list(Eqn)
    """
    import parse
    import yaml

    with open(fp, "r") as f:
        data = yaml.load(f, Loader=yaml.Loader)

    # reaction list is a dict
    # keys are a reaction ID, values are the reaction string
    # in the RACM2 paper they use R1, R2, etc. though, not IRR_1, IRR_2

    # format of the reaction strings is like
    #   NO2 ->[j] O3P + NO
    #   NO3 + HO ->[k] HO2 + NO2
    # only put type if not `s`
    rxn_pattern = r"{lhs} ->[{ktype}] {rhs}"  # for the parse module

    # ! IRR_333 is skipped on loading for some reason?
    # no it is just out of order.
    # but one num is being skipped (233, it was 333) but I fixed

    eqns = []
    # for j, (k, v) in enumerate(data["reaction_list"].items()):
    for j in range(350):  # there *seem* to be 350 rxns in the file

        k = f"IRR_{j+1}"
        v = data["reaction_list"].get(k)

        if v is None:
            warnings.warn(f"key {k} doesn't exist in the file")
            continue

        # print(j, k)
        # print(v)

        res = parse.parse(rxn_pattern, v)
        # print(f"  {res}")

        if not res["rhs"]:
            raise Exception("there should be a RHS for every reaction")

        lhs = res["lhs"]
        rhs = res["rhs"]

        # reactants
        ms_rcts = parse_eqn_side(lhs)  # reactant (mult, spc)

        # producst
        ms_pdts = parse_eqn_side(rhs)  # product (mult, spc)

        # form equation
        eqn = Eqn(ms_rcts, ms_pdts, k=1.0, name=f"R{j+1}")

        eqns.append(eqn)

    return eqns
