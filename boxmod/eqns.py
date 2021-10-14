"""
Mechanism object (`EqnSet`) and analysis tools.
"""
import subprocess
from collections import defaultdict
from collections import namedtuple

import numpy as np

__all__ = (
    "EqnElement",
    "Eqn",
    "EqnSet",
)


# TODO: define add/subtract for EqnElement and Eqn ?

# TODO: try using attrs for some of these classes (or dataclass / namedtuple even)


class EqnElement:
    """A single reactant or product, with stoichiometric multiplier."""

    def __init__(self, mult, spc):
        """
        Parameters
        ----------
        mult : {int, float}
            Stoichiometric multiplier for `spc` in the reaction.
        spc : str
            Species identifier,
            e.g., "H2O", or "ISO".
        """
        self.mult = float(mult)
        self.spc = spc
        #

    def _to_string(self, keep_mult_unity=True):
        if not keep_mult_unity and self.mult == 1.0:
            return f"{self.spc}"
        else:
            return f"{self.mult:.3g} {self.spc}"

    def __str__(self):
        return self._to_string()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__str__()})"

    def __eq__(self, other):
        return self.mult == other.mult and self.spc == other.spc


# TODO: optional description for EqnElement? latex code (for mhchem or chemformula)?


# only normal chemical formulas get the auto-formatting
LATEX_SPC_MAP = {
    "O1D": "O(^1D)",
    "O3P": "O(^3P)",
}


class Eqn:
    """A full set of equation elements."""

    def __init__(self, reactants, products, k, *, str_orig=None, name=None, k_fn=None):
        """
        reactants : list of (mult, spc) tuples
            to initialize `EqnElement`s
        products : " "
            " "
        k : float
            a sample/default value to use for the reaction rate coeff
            units: e.g., s^-1, cm^3 molec^-1 s^-1

        Optional
        --------
        str_orig : str, optional
            original string that this equation was parsed from (for later reference)
        name : str, optional
            ID for the equation, e.g., "R22"
        k_fn : Callable, optional
            Function to compute k based on T, M, etc.
        """
        self.rct = [EqnElement(mult, spc) for (mult, spc) in reactants]
        self.pdt = [EqnElement(mult, spc) for (mult, spc) in products]
        self.k = k
        #
        self.str_orig = str_orig
        self.name = name
        self.k_fn = k_fn

        # TODO: check if species are repeated, change to mult form. with warning?
        # TODO: optional canceling of species (if on both sides)?
        # TODO: optional atom/mass balance checking?

        # private lists of species in the reaction
        self._rct_spc = [e[1] for e in reactants]
        self._pdt_spc = [e[1] for e in products]
        self._all_spc = self._rct_spc + self._pdt_spc

    def rate_expr(self):  # functional form of rate coeff
        raise NotImplementedError

    def rate_expr_str(self):
        raise NotImplementedError

    def eqn_str(self, sign="=", keep_mult_unity=True):
        """Just equation/reaction (no k)

        keep_mult_unity : bool
            whether to print multipliers (stoichiometric coeffs) equal to 1.0
        """
        rct_s = " + ".join(r._to_string(keep_mult_unity=keep_mult_unity) for r in self.rct)
        pdt_s = " + ".join(p._to_string(keep_mult_unity=keep_mult_unity) for p in self.pdt)
        return f"{rct_s} {sign} {pdt_s}"

    # TODO: rate_str here? (generalized form, like `k[A][B]`)

    def __str__(self):
        return f"{self.eqn_str()} : {self.k:.3g}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__str__()})"

    def __eq__(self, other):
        # compare all equation elements
        ees_self = self.rct + self.pdt
        ees_other = other.rct + other.pdt
        return all(ee_s == ee_o for ee_s, ee_o in zip(ees_self, ees_other))

    def __hash__(self):
        # make hashable so can take `set()` of a list of Eqn
        return hash(repr(self))

    def to_latex(self, *, chem="mhchem"):
        """Equation string latex-style.

        chem : str
            'mhchem' or 'chemformula'
            mhchem syntax is recognized in MathJax/KaTeX

        """
        cmd = {"mhchem": r"\ce", "chemformula": r"\ch"}.get(chem)
        if cmd is None:
            raise ValueError(f"`chem` must be 'mhchem' or 'chemformula' not {chem!r}")

        eqn = self.eqn_str(sign="->", keep_mult_unity=False)

        # make replacments
        for orig, new in LATEX_SPC_MAP.items():
            eqn = eqn.replace(
                orig,
                new,
            )

        return f"{cmd}{{{eqn}}}"

    def to_text(self):
        # pretty much same as the latex one
        eqn = self.eqn_str(sign="->", keep_mult_unity=False)
        return eqn


# ? use Pandas DataFrame as container instead?


class EqnSet:
    """A set of chemical equation---a chemical mechanism."""

    def __init__(self, equations, species=None, *, name="eqnset", long_name="A mechanism"):
        """
        equations : list(Eqn)

        species : list(str)
            IDs of the species to compute rates of change for
            species with variable conc. (i.e., not O2, N2, etc.)

        Optional
        --------
        name : str
            identifier for the mechanism
        long_name : str
            longer description
        """
        # TODO: EqnList as a separate object, with custom iterator stuff?
        # TODO: pass fixed species instead? then species as all - fixed

        eqn_all_spc = self._find_all_spc(equations)
        if species is not None:
            # check provided species list against the species in the reactions
            for spc in species:
                if spc not in eqn_all_spc:
                    raise ValueError(f"species {spc} not found in any of the reactions.")
        else:
            species = eqn_all_spc
        self.spcs = species

        # set equations using the setter method
        # which performs the mechanism analysis in full
        self.eqns = equations

        #
        self.name = name
        self.long_name = long_name

    def __str__(self):
        return f"[{'; '.join(str(eqn) for eqn in self.eqns)}]"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__str__()})"

    def _find_all_spc(self, eqns=None):
        "Find all spc in internal or external equation list"
        # TODO: option to check LHS of equations only
        # external option so that we can run this before setting `self.eqns`
        # which needs `self.spcs` to be set
        if eqns is None:
            eqns = self.eqns
        all_spc_raw = [spc for e in eqns for spc in e._all_spc]

        spc_unique_case = set(all_spc_raw)
        # spc_unique_no_case = set(spc.upper() for spc in all_spc_raw)
        # TODO: check for possible case mistakes in species
        # likely_dupes = spc_unique_case - spc_unique_case

        return sorted(spc_unique_case)

    @property
    def eqns(self):
        return self._eqns

    @eqns.setter
    def eqns(self, eqn_list):
        self._eqns = eqn_list
        # TODO: check for dupe equations

        # form the KPP S matrices
        self._analyze_eqns()

        # TODO: set up omega: list of reaction velocities

        # TODO: other prep/validation stuff?

    @property
    def n_spc(self):
        return len(self.spcs)

    @property
    def n_rxn(self):
        return len(self.eqns)

    def _analyze_eqns(self):
        """Find contributors to each species's dCdt
        and preserve in the S matrix (S for stoich.).

        S: spc (rows) x rxn (cols)

        s_ij is the number (stoichiometric coeff) of species *i*
             that are lost/gained in reaction *j*

        S_p: S+ (produced)
        S_m: S- (destroyed)


        Matrix multiplication of S and lamba (n_rxn; the vector of reaction rates)
        will give the vector of dCdt values

        This fn is called when the equation list `self.eqns` is changed.

        Notation based on KPP.
        """

        n_spc = self.n_spc
        n_rxn = self.n_rxn

        # to look up index by species ID
        spc_ind = {s: i for i, s in enumerate(self.spcs)}

        S = np.zeros((n_spc, n_rxn))
        S_p = np.zeros((n_spc, n_rxn))
        S_m = np.zeros((n_spc, n_rxn))

        # also collect the reaction inds that affect a certain reaction (+ or -)
        spc_rxn_inds = defaultdict(list)
        # and the indices of the reactants
        rxn_spc_inds = defaultdict(list)

        # collect species connections (somewhere else? since not needed for integration)
        # spc_reacts_with = defaultdict(list)
        # spc_produces = defaultdict(list)
        # spc_produced_by = defaultdict(list)

        for j, eqn in enumerate(self.eqns):

            # TODO: Eqn properties should end in `s` for consistency
            # TODO: avoid extra looping?
            # reactants (consumed: -1*mult)
            # spc_rcts = [r.spc for r in eqn.rct]
            for r in eqn.rct:
                i = spc_ind[r.spc]
                S_m[i, j] = 1 * r.mult

                spc_rxn_inds[r.spc].append(j)

                rxn_spc_inds[j].append(i)

                # spc_reacts_with[r.spc].append()

            # products (produced: +1*mult)
            # spc_pdts = [p.spc for p in eqn.pdt]
            for p in eqn.pdt:
                i = spc_ind[p.spc]
                S_p[i, j] = 1 * p.mult

                # note: will be adding twice if spc is on both sides of the reaction
                spc_rxn_inds[p.spc].append(j)

        # hack to ensure spc_rxn_inds has unique indices only
        for k in spc_rxn_inds:
            spc_rxn_inds[k] = list(set(spc_rxn_inds[k]))

        # S from S+ and S- (both contain positive numbers only)
        S = S_p - S_m

        # set class attrs
        self.S = S
        self.S_m = S_m
        self.S_p = S_p
        self.spc_rxn_inds = spc_rxn_inds
        self.rxn_spc_inds = rxn_spc_inds

    def print_rate_strings(self, *, rate_format="R", show_zeros=False):
        """Generate generalized (no k expressions) rate expressions
        for each species.

        Parameters
        ----------
        rate_format : str
            `'R'`: abbreviated format, showing reaction numbers

            `'k'`: more detailed format, showing numbered rate coeffs and concentrations
        show_zeros : bool
            Whether to show non-contributing reaction rates in the dcdt expressions.
            Mostly for debugging (there shouldn't be any zeros unless species is on both sides
            of an equation).

        For quick validation by eye.
        """

        for i, spc in enumerate(self.spcs):

            s_i = self.S[i, :]  # rxn stoich coeffs for species i

            rxn_inds = self.spc_rxn_inds[spc]

            s_rxns = []
            count = 0  # counter that won't incremement if skip (for 0s)
            for ind in rxn_inds:  # reaction indices: `j` in the KPP notation
                mult = s_i[ind]

                if mult == 0.0 and not show_zeros:
                    continue

                sign = "+" if mult > 0 else "-"
                abs_mult = abs(mult)

                if count > 0 or sign == "-":  # not first reaction or negative (need to show)
                    pref = f"{sign} "
                else:
                    pref = ""

                # only show stoich if not 1
                s_ij = f"{abs_mult:.3g} " if abs_mult != 1.0 else ""

                # reaction rate itselfs
                if rate_format == "R":
                    rate = f"R{ind+1}"

                elif rate_format == "k":

                    # // rate_spc = [self.spcs[spc_] for spc_ in self.rxn_spc_inds[ind]]
                    # // s_rate_spc = "".join(f"[{spc_}]" for spc_ in rate_spc)

                    # TODO: factor out
                    s_rate_spc_parts = []
                    for i_2 in self.rxn_spc_inds[ind]:
                        power = self.S_m[i_2, ind]
                        rate_spc = self.spcs[i_2]
                        s_pow = f"^{power:.3g}" if power != 1.0 else ""
                        s_rate_spc_parts.append(f"[{rate_spc}]{s_pow}")

                    s_rate_spc = "".join(s_rate_spc_parts)

                    rate = f"k{ind+1}{s_rate_spc}"
                else:
                    raise ValueError(f"invalid `format` {rate_format!r}")

                s_rxn = f"{pref}{s_ij}{rate}"

                s_rxns.append(s_rxn)
                count += 1

            s_rxns = " ".join(s_rxns) if s_rxns else "0"

            print(f"d[{spc}]/dt = {s_rxns}")

    def print_latex(self):
        """Print LaTeX block."""
        print(r"\begin{center}" "\n" r"\begin{tabular}{ r l }")

        for i, eqn in enumerate(self.eqns):
            print(rf"R{i+1} & {eqn.to_latex(chem='chemformula')} \\")

        print(r"\end{tabular}" "\n" r"\end{center}")

    # def _rxn_rate(self, rxn_ind, rate_format="k"):
    #     """Rate expression for the reaction with index `rxn_ind`.

    #     """

    def rxn_rates(self, c, *, ret_code=False):
        """Compute reaction rates (using current conc. vector `c`).

        ret_code : bool
            True: return string (list of?) of Python code
            False: return vector of reaction rates
                as a column vector!
        """
        # TODO: k calculations (for now using the constant k's)
        if ret_code:
            raise NotImplementedError

        rates = np.zeros((self.n_rxn, 1))
        for j, eqn in enumerate(self.eqns):

            # note: could instead do this vectorized, with `np.prod`?
            rate_j = eqn.k
            for i in self.rxn_spc_inds[j]:  # look up the species involved
                power = self.S_m[i, j]
                rate_j *= c[i] ** power

            rates[j] = rate_j

        return rates

    def dcdt_fn(self, c, *, fixed=None, sources=None):
        """Calculate dcdt given concentrations `c`.
        This can be used with `scipy.integrate.solve_ivp`.

        Parameters
        ----------
        c : array_like
            Concentrations of species (molec cm^-3).
        fixed : dict
            Fixed rates (molec cm^-3 s^-1), used to override the calculated dcdt.
        sources : dict
            Additional constant source/sink terms to add on (molec cm^-3 s^-1). For sink term,
            just make it negative.

        Returns
        -------
        dcdt : array_like
            column vector of conc. tendencies (molec cm^-3 s^-1)
        """

        S = self.S
        omega = self.rxn_rates(c)  # this is KPP notation
        dcdt = S @ omega  # inner product: gives a column vector with n_spc rows

        # adjust for fixed rates
        if fixed is not None:
            for spc, r in fixed.items():
                dcdt[self.spcs.index(spc)] = r

        # sources/sinks
        if sources is not None:
            for spc, r in sources.items():
                dcdt[self.spcs.index(spc)] += r

        return dcdt

    # TODO: Jacobian

    def gen_dcdt_fn_code(self):
        """Generate Python code for a dcdt fn."""
        raise NotImplementedError
        # need functions for the rates
        # some rates need a k function

    def plot_graph(self, which="r-p", *, lib="graphviz"):
        """Plot graph of connections between species.

        which : str
            r-p: reactant-product conections (for species i, the species on other side of reaction)
            co-r: co-reactants (reacts with)
            co-p: co-products (produced alongside)

        lib : str {'graphviz', 'networkx'}
            the graph plotting library to use

        """
        plotters = {
            "graphviz": self._plot_graph_graphviz,
            "networkx": self._plot_graph_networkx,
        }
        plotter = plotters.get(lib, None)
        if plotter is None:
            raise ValueError(f"invalid `lib` choice {lib!r}")

        if which == "r-p":
            cons = self._find_graph_connections()
            graphviz_kw = {}  # strict=False by default
            # graphviz_kw = {"strict": True}  # merge multiples of the same connection (edge)
            general_kw = {
                "title": "Product-reactant connections (i.e., across the reaction arrow)",
            }
        elif which in ["co-r", "co-p"]:
            cons = self._find_graph_connections_co(
                which={"co-r": "reactants", "co-p": "products"}[which],
            )
            graphviz_kw = {}
            # graphviz_kw = {"strict": True}
            general_kw = {
                "title": "Co-reactants" if which == "co-r" else "Co-products",
            }
        else:
            raise ValueError(f"invalid `which` choice {which!r}")

        general_kw.update({"which": which})  # for part of saved fig name

        # invoke selected plotter
        kwargs = {**graphviz_kw, **general_kw}
        plotter(cons, **kwargs)  # hack for now (graphviz only)

    # TODO: directed graph (reactant->products) option
    # TODO: co-reactants (reacts with) option (but how to deal with photo-reactions, self-reactions?, decomp, etc.)

    def _find_graph_connections(self):
        """For species, find the species that are on the other side of reaction equations.

        Returns
        -------
        list of named tuples with
            spc
            rxn_name
        """
        cons = defaultdict(list)
        Con = namedtuple("Con", "spc rxn_name")
        for spc in self.spcs:
            rxn_inds = self.spc_rxn_inds[spc]
            for j in rxn_inds:
                eqn = self.eqns[j]
                rcts_j = eqn._rct_spc
                pdts_j = eqn._pdt_spc

                # record all opposite
                # spc_con_j = rcts_j if spc in pdts_j else pdts_j

                # only record product, where spc is a reactant
                if spc in rcts_j:
                    spc_con_j = pdts_j
                else:
                    continue

                # TODO allow picking between these two (up/dn) as an option
                # and determine if they truly give different results, or just different orders
                # (i.e., collect all A->Bs from both methods and sort them)

                # only record reactants, where spc is a product
                # if spc in pdts_j:
                #     spc_con_j = rcts_j
                # else:
                #     continue

                # the graph edge needs a label
                if eqn.name is not None:
                    eqn_name = eqn.name
                else:
                    eqn_name = f"R{j+1}"

                con_j = [Con(s, eqn_name) for s in spc_con_j]

                cons[spc].extend(con_j)

            # uniques only?
            # we wouldn't want that if trying to capture all reactions
            # cons[spc] = list(set(con[spc]))

        return cons

    def _find_graph_connections_co(self, which="reactants"):
        """Find co-reactants or -products.

        (As opposed to the species on other side of the reaction arrow.)

        which : str {'reactants', 'products'}

        Returns
        -------
        list of named tuples with
            spc
            rxn_name
        """
        if which not in ["reactants", "products"]:
            raise ValueError(f"invalid `which` {which!r}")

        cons = defaultdict(list)
        cons_spc = defaultdict(list)
        Con = namedtuple("Con", "spc rxn_name")
        for spc in self.spcs:
            rxn_inds = self.spc_rxn_inds[spc]
            for j in rxn_inds:
                # get reaction info
                eqn = self.eqns[j]
                rcts_j = eqn._rct_spc
                pdts_j = eqn._pdt_spc

                if which == "reactants":
                    if spc not in rcts_j:  # spc is not a reactant (is a product)
                        continue  # go to next equation
                    to_filter = rcts_j

                else:  # products
                    if spc not in pdts_j:  # spc is not a product (is a reactant)
                        continue
                    to_filter = pdts_j

                # reaction needs a label
                if eqn.name is not None:
                    eqn_name = eqn.name
                else:
                    eqn_name = f"R{j+1}"

                # check for existent opposite connections
                # (we don't want to have both A->B and B->A for the same reaction)
                to_filter = [spc_b for spc_b in to_filter if spc not in cons_spc[spc_b]]

                cons_spc[spc].extend(to_filter)  # species strings only

                # if to_filter:
                # spc_con_j = [s for s in to_filter if s != spc]
                con_j = [Con(s, eqn_name) for s in to_filter if s != spc]
                # ? ^ is this the most efficient way to do this?

                cons[spc].extend(con_j)

            # only need unique reactions
            cons[spc] = list(set(cons[spc]))
            # should already be unique unless a spc is in rcts/pdts multiple times
            # TODO: should do some checks for this
            # ^ but we still get both A->B and B->A

        return cons

    # TODO: this only needs the graph connections, could remove from the class

    def _plot_graph_graphviz(self, cons, *, strict=False, title="", which=""):
        from graphviz import Graph

        # import tempfile

        # can also set global node_attr and edge_attr here
        dot = Graph(
            comment="Species connections",
            node_attr={
                "fillcolor": "blue",
                "color": "blue",
                "fontcolor": "blue",
                "fontsize": "14",  # should be the default
            },
            edge_attr={
                "color": "grey",
                "fontsize": "12",
            },
            graph_attr={
                "labelloc": "t",
                "label": f"{title}\nfor mechanism: {self.long_name}",
                "labelfontsize": "18",  # doesn't seem to have an effect
            },
            strict=strict,
        )

        # add nodes
        for spc in self.spcs:
            dot.node(spc)

        # add edges
        for spc in self.spcs:
            # we can add all at once like this, but didn't seem to allow specifying label
            # dot.edges([(spc, con.spc) for con in cons[spc]])
            for con in cons[spc]:
                dot.edge(spc, con.spc, label=con.rxn_name)  # , _attributes=edge_attrs)

        # render
        # fp = tempfile.TemporaryFile(suffix=".gv", delete=False)
        # fp.close()
        # dot.render(fp.name, view=True)
        fn = f"{self.name}_{which}.gv"
        formats = ["pdf", "png", "svg"]
        for fmt in formats:
            try:
                dot.render(fn, quiet=True, format=fmt)
            except subprocess.CalledProcessError:  # as e:
                # warnings.warn(str(e))
                pass
        # ^ raises CalledProcessError saying it can't find `dot.bat`,
        # but in fact it is there in Path (and still works). Maybe a cmd/PowerShell issue

    def _plot_graph_networkx(self, cons):
        raise NotImplementedError
        # import networkx as nx
        # G = nx.Graph()
