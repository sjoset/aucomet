#!/usr/bin/env python3

import sys
import re
import pathlib
import numpy as np
import pandas as pd

from typing import List, Optional
from dataclasses import dataclass


"""Uses information downloaded from the SWIFT database at swift.co.uk to identify conventional comet names based on the target name typed in by the operator"""


@dataclass
class CometNameRegex:
    """Holds the results of searching 'original_name' for patterns that match long-period, short-period, and interstellar comet names"""

    original_name: str
    long_period_name: Optional[str]
    short_period_name: Optional[str]
    interstellar_name: Optional[str]


@dataclass
class ConventionalCometName:
    """Holds the results after examining a CometNameRegex and deciding which results to use to"""

    original_name: str
    conventional_name: str


def match_long_period_name(comet_name: str) -> Optional[str]:
    """Searches comet_name for anything resembling a long-period comet naming convention"""
    # match naming convention of long period comets: C/[YEAR][one optional space character][one to two letters][one to two digits]
    long_period_match = re.search("C/[0-9]{4}\\s?[A-Z]{1,2}[0-9]{1,2}", comet_name)
    if long_period_match:
        # get the matched part of the string if there was a match
        long_period_name = long_period_match.group() if long_period_match else None
    else:
        # try the same match but instead of "C/...", look for "Comet..."
        long_period_match = re.search(
            "Comet[0-9]{4}\\s?[A-Z]{1,2}[0-9]{1,2}", comet_name
        )
        if long_period_match:
            long_period_name = re.sub("Comet", "C/", long_period_match.group())
        else:
            long_period_name = None

    return long_period_name


def match_short_period_name(comet_name: str) -> Optional[str]:
    """Searches comet_name for anything resembling a short-period comet naming convention"""
    # try to find  P/[YEAR][one to two characters][one to three numbers]
    short_period_match = re.search("P/[0-9]{4}[A-Z]{1,2}[0-9]{1,3}", comet_name)
    if short_period_match:
        short_period_name = short_period_match.group() if short_period_match else None
    else:
        # match naming convention of short period comets: [1 to 3 digits]P
        short_period_match = re.search("[0-9]{1,3}P", comet_name)
        if short_period_match:
            short_period_name = short_period_match.group()
        else:
            short_period_name = None

    return short_period_name


def match_interstellar_name(comet_name: str) -> Optional[str]:
    """Searches comet_name for anything resembling an interstellar comet naming convention"""
    # try interstellar interstellar naming scheme: we include the slash on the end so we remove that later
    interstellar_match = re.search("[0-9]{1,3}I/", comet_name)
    interstellar_name = interstellar_match.group()[:-1] if interstellar_match else None

    return interstellar_name


def extract_comet_name(comet_name: str) -> CometNameRegex:
    """
    Takes a string, searches for comet naming formats, and returns a CometNameRegex with the results
    """

    return CometNameRegex(
        original_name=comet_name,
        long_period_name=match_long_period_name(comet_name),
        short_period_name=match_short_period_name(comet_name),
        interstellar_name=match_interstellar_name(comet_name),
    )


def count_extracted_names(cnr: CometNameRegex) -> int:
    return sum(
        [
            1 if cnr.long_period_name else 0,
            1 if cnr.short_period_name else 0,
            1 if cnr.interstellar_name else 0,
        ]
    )


def has_naming_conflict(cnr: CometNameRegex) -> bool:
    return count_extracted_names(cnr) > 1


def is_unnamed(cnr: CometNameRegex) -> bool:
    return count_extracted_names(cnr) == 0


def is_long_period(cnr: CometNameRegex) -> bool:
    return (
        cnr.long_period_name is not None
        and cnr.short_period_name is None
        and cnr.interstellar_name is None
    )


def is_short_period(cnr: CometNameRegex) -> bool:
    return (
        cnr.long_period_name is None
        and cnr.short_period_name is not None
        and cnr.interstellar_name is None
    )


def is_interstellar(cnr: CometNameRegex) -> bool:
    return (
        cnr.long_period_name is None
        and cnr.short_period_name is None
        and cnr.interstellar_name is not None
    )


def count_comet_name_conflicts(cnrs: List[CometNameRegex]) -> int:
    return sum([1 if has_naming_conflict(cnr) else 0 for cnr in cnrs])


def count_unnamed_comets(cnrs: List[CometNameRegex]) -> int:
    return sum([1 if is_unnamed(cnr) else 0 for cnr in cnrs])


def count_long_period_comets(cnrs: List[CometNameRegex]) -> int:
    return sum([1 if is_long_period(cnr) else 0 for cnr in cnrs])


def count_short_period_comets(cnrs: List[CometNameRegex]) -> int:
    return sum([1 if is_short_period(cnr) else 0 for cnr in cnrs])


def count_interstellar_comets(cnrs: List[CometNameRegex]) -> int:
    return sum([1 if is_interstellar(cnr) else 0 for cnr in cnrs])


def attempt_manual_fix(comet_name: str) -> str:
    """
    Some of the swift names contain enough information to identify the comet,
    but not in the right format for a regular expression, so we provide the conventional name here
    """
    name_fixes = {
        "CometCatalinaOrbit1": "C/2013US10",
        "CometCatalinaOrbit2": "C/2013US10",
        "CometCatalinaOrbit3": "C/2013US10",
        "CometCatalinaOrbit4": "C/2013US10",
        "C/2013USCatalina": "C/2013US10",
        "C/McNaught 2006 P1": "C/2006P1",
        "CometC/2012(ISON)": "C/2012S1",
        "CometC1/2013SidingSprings": "C/2013A1",
        "CometC2012S1(ISON)": "C/2012S1",
        "CometGarradd": "C/2009P1",
        "CometHolmes": "17P",
        "Comet Holmes": "17P",
        "CometLulin": "C/2007N3",
        "CometVL62": "C/2015VL62",
    }
    return name_fixes.get(comet_name, "Unnamed")


def decide_comet_name(cnr: CometNameRegex) -> ConventionalCometName:
    """Given results in a CometNameRegex, decide how to resolve conflicts if we detect multiple possible names"""
    if is_unnamed(cnr):
        return ConventionalCometName(
            original_name=cnr.original_name,
            conventional_name=attempt_manual_fix(cnr.original_name),
        )
    elif has_naming_conflict(cnr):
        if cnr.long_period_name:
            # the long period name is very restrictive to match, so if one is found,
            # ignore the other matches as false positives
            return ConventionalCometName(
                original_name=cnr.original_name,
                conventional_name=str(cnr.long_period_name),
            )
        else:
            # otherwise, the new name will just show the conflict for resolution later
            return ConventionalCometName(
                original_name=cnr.original_name,
                conventional_name=f"{cnr.original_name}"
                + " ====> Long: "
                + str(cnr.long_period_name)
                + " ====> Short: "
                + str(cnr.short_period_name)
                + " ====> Interstellar: "
                + str(cnr.interstellar_name),
            )
    else:
        return ConventionalCometName(
            original_name=cnr.original_name,
            conventional_name=(cnr.long_period_name if cnr.long_period_name else "")
            + (cnr.short_period_name if cnr.short_period_name else "")
            + (cnr.interstellar_name if cnr.interstellar_name else ""),
        )


def print_summary(cnrs: List[CometNameRegex], comet_dataframe: pd.DataFrame) -> None:
    comet_count = len(cnrs)
    confict_count = count_comet_name_conflicts(cnrs)
    unnamed_count = count_unnamed_comets(cnrs)
    success_count = comet_count - unnamed_count
    print(f"Unambiguously named: {comet_count - confict_count - unnamed_count}")
    print(f"Possible conflicts: {confict_count}")
    print(f"To be manually fixed: {unnamed_count}")
    print(
        f"Success assuming conflicts are false positives: {success_count}/{comet_count} ({100 * (success_count)/comet_count:4.1f} %)"
    )
    print("")

    long_period_count = count_long_period_comets(cnrs)
    short_period_count = count_short_period_comets(cnrs)
    interstellar_count = count_interstellar_comets(cnrs)
    print("Comet observations with successful extractions:")
    print(f"\tLong period: {long_period_count}")
    print(f"\tShort period: {short_period_count}")
    print(f"\tInterstellar: {interstellar_count}")
    print("")

    # if there are multiple results, we take the long period result, so combine the conflicts with long-period results
    rename_long_period_comets = [
        decide_comet_name(cnr).conventional_name
        for cnr in cnrs
        if (is_long_period(cnr) or has_naming_conflict(cnr))
    ]
    renamed_long_period_set = set(rename_long_period_comets)

    short_period_comets = [
        decide_comet_name(cnr).conventional_name for cnr in cnrs if is_short_period(cnr)
    ]
    short_period_set = set(short_period_comets)

    interstellar_comets = [
        decide_comet_name(cnr).conventional_name for cnr in cnrs if is_interstellar(cnr)
    ]
    interstellar_set = set(interstellar_comets)

    unnamed_comets = [decide_comet_name(cnr) for cnr in cnrs if is_unnamed(cnr)]
    manually_renamed_set = set(
        [
            (x.original_name, x.conventional_name)
            for x in unnamed_comets
            if x.conventional_name != "Unnamed"
        ]
    )
    not_manually_renamed = [
        x.original_name for x in unnamed_comets if x.conventional_name == "Unnamed"
    ]

    print(f"Detected long period comets ({len(renamed_long_period_set)}):")
    for x in sorted(renamed_long_period_set):
        print(f"\t{x}")
    print("")

    print(f"Detected short period comets ({len(short_period_set)}):")
    for x in sorted(short_period_set):
        print(f"\t{x}")
    print("")

    print(f"Detected interstellar comets ({len(interstellar_set)}):")
    for x in sorted(interstellar_set):
        print(f"\t{x}")
    print("")

    print(f"Manually renamed entries ({len(manually_renamed_set)}):")
    for x in sorted(manually_renamed_set, key=lambda x: x[0]):
        print(f"\t{x[0]:>30} ====> {x[1]}")
    print("")

    print(f"Entries not renamed: {len(not_manually_renamed)}")
    for x in sorted(set(not_manually_renamed)):
        print(f"\t{x} ({not_manually_renamed.count(x)})")
    print("")

    problem_obsids = []
    for x in set(not_manually_renamed):
        df = comet_dataframe.loc[comet_dataframe.original_swift_name == x]
        problem_obsids += [x for x in df["obsid"].to_string(index=False).split()]

    print(f"Problem observation ids: {len(problem_obsids)}")
    for p in sorted(problem_obsids):
        print(f"{p:>010}")
    print("")

    print(
        f"Total renaming coverage: {comet_count - len(not_manually_renamed)}/{comet_count} ({100*(comet_count - len(not_manually_renamed))/comet_count}%)"
    )
    print("")


def read_swift_database_csv(fname: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(filepath_or_buffer=fname)  # type: ignore


def write_swift_database_csv(
    fname: pathlib.Path, comet_dataframe: pd.DataFrame
) -> None:
    comet_dataframe.to_csv(fname, index=False)


def main():
    # read our data
    comet_dataframe = read_swift_database_csv(pathlib.Path("swift_comet_database.csv"))
    raw_comet_names = comet_dataframe["original_swift_name"]

    # try to match the observation name to a conventional comet name
    comet_name_regexes = list(map(extract_comet_name, raw_comet_names))  # type: ignore

    print_summary(cnrs=comet_name_regexes, comet_dataframe=comet_dataframe)

    # produce a list of (observation name, new name)
    conventional_names = list(map(decide_comet_name, comet_name_regexes))

    print("Random selection of 10 renamings:")
    for ncn in np.random.choice(np.array(conventional_names), size=10):
        print(f"{ncn.original_name:>40} ===> {ncn.conventional_name}")

    # add our names to the dataframe
    comet_dataframe.insert(
        loc=2,
        column="conventional_names",
        value=[cn.conventional_name for cn in conventional_names],
    )

    # write out the results
    write_swift_database_csv(
        fname=pathlib.Path("swift_comet_database_normalized.csv"),
        comet_dataframe=comet_dataframe,
    )


if __name__ == "__main__":
    sys.exit(main())
