"""
Discussion and limits analysis module.

Provides granular helpers for Section 4 (Limits of Identifying the
Argument Structure in Searle's Texts Using NLI):
- 4.1 Limits of labeling argument types
- 4.2 Need for argument mining
- 4.3 Need for probabilistic analysis
- 4.4 Need for justification
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

from nli_analysis import (
    predict_nli,
    define_mbp_argument,
    define_cra_argument,
    define_larger_argument,
)


# --- 4.1 helpers ---

def show_entailment_cases(combined_df):
    """Return rows from combined_df where the human prediction is 'entailment'."""
    return combined_df[combined_df["Human Prediction"] == "entailment"]


# --- 4.2 helpers ---

def plot_nli_complexity(max_n=20):
    """Plot the growth of possible NLI comparisons vs 2^n.

    Shows why exhaustive NLI analysis is infeasible for texts
    with many sentences.
    """
    def nli_exe(n):
        total = 0
        for k in range(1, n):
            total += comb(n, k) * comb(n - k, 1)
        return total

    def two_power(n):
        return 2 ** n

    n_values = np.arange(1, max_n + 1)
    nli_exe_values = [nli_exe(n) for n in n_values]
    two_power_values = [two_power(n) for n in n_values]

    plt.figure(figsize=(10, 6))
    plt.plot(
        n_values, nli_exe_values,
        label=r"$\sum_{k=1}^{n-1} \binom{n}{k} \cdot \binom{n-k}{1}$",
        marker="o",
    )
    plt.plot(n_values, two_power_values, label=r"$2^n$", marker="x")
    plt.xlabel("n")
    plt.ylabel("Value")
    plt.title(r"$2^n$ and the number of different NLI for $n$ sentences")
    plt.legend()
    plt.grid(True)
    plt.show()


def print_conclusions():
    """Print the conclusion strings for CRA, LARG, and MBP."""
    _, conc_cra = define_cra_argument()
    _, conc_larg = define_larger_argument()
    _, conc_mbp = define_mbp_argument()
    print(conc_cra)
    print(conc_larg)
    print(conc_mbp)


# --- 4.3 helpers ---

def example_boy_ball(nli_model):
    """Boy hits a ball / playing baseball example from Chen et al. (2020)."""
    p1 = "A boy hits a ball with a bat."
    h1 = "The boy is playing in a baseball game."
    print(predict_nli(nli_model, p1, h1))


def example_room_mbp_partial(nli_model):
    """MBP single-premise (room_mbp) -> conclusion (with com_mbp omitted)."""
    room_mbp = (
        "As far as the Chinese is concerned, I simply behave like a computer. "
        "I have inputs and outputs that are indistinguishable from those of "
        "the native Chinese speaker, but I still understand nothing."
    )
    conc_mbp = (
        "Computer understands nothing of any stories, whether in Chinese, "
        "English, or whatever."
    )
    print(predict_nli(nli_model, room_mbp, conc_mbp))


def example_larg_partial(nli_model):
    """LARG with 'prog' + 'suf' (omitting 'mind') -> conclusion."""
    prog = "Implemented programs are by definition purely formal or syntactical."
    suf = "Syntax is not by itself sufficient for, nor constitutive of, semantics."
    conc_larg = "Implemented programs are not constitutive of minds."
    print(predict_nli(nli_model, prog + " " + suf, conc_larg))


# --- 4.4 helpers ---

def example_quine(nli_model):
    """'Quine is married' vs 'Quine is a bachelor' - clear contradiction."""
    p2 = "Quine is married."
    h2 = "Quine is a bachelor."
    print(predict_nli(nli_model, p2, h2))


def example_larg_orderings(nli_model):
    """LARG with mind+prog and prog+mind premise orderings."""
    mind = "Minds have mental or semantic contents."
    prog = "Implemented programs are by definition purely formal or syntactical."
    conc_larg = "Implemented programs are not constitutive of minds."
    print(predict_nli(nli_model, mind + " " + prog, conc_larg))
    print(predict_nli(nli_model, prog + " " + mind, conc_larg))


def example_intermediate(nli_model):
    """Examples showing the role of intermediate premises in justification."""
    mind = "Minds have mental or semantic contents."
    prog = "Implemented programs are by definition purely formal or syntactical."
    conc_larg = "Implemented programs are not constitutive of minds."
    inter = "Programs with minds are mental and semantic."
    inter2 = "Minds with programs are mental and semantic."

    print(predict_nli(nli_model, mind + " " + prog, conc_larg))
    print()
    print(predict_nli(nli_model, mind + " " + prog, inter))
    print(predict_nli(nli_model, inter, conc_larg))
    print()
    print(predict_nli(nli_model, mind + " " + prog, inter2))
    print(predict_nli(nli_model, inter2, conc_larg))
