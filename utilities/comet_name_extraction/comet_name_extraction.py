#!/usr/bin/env python3

import sys
import re
import numpy as np
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class CometNameRegex:
    original_name: str
    long_period_name: Optional[str]
    short_period_name: Optional[str]
    interstellar_name: Optional[str]


@dataclass
class CometRenaming:
    original_name: str
    new_name: str


def extract_comet_name(comet_name: str) -> CometNameRegex:
    """
    Takes a string, searches for comet naming formats, and extracts it into the CometNameRegex dataclass
    """
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

    # try interstellar interstellar naming scheme: we include the slash on the end so we remove that later
    interstellar_match = re.search("[0-9]{1,3}I/", comet_name)
    interstellar_name = interstellar_match.group()[:-1] if interstellar_match else None

    return CometNameRegex(
        original_name=comet_name,
        long_period_name=long_period_name,
        short_period_name=short_period_name,
        interstellar_name=interstellar_name,
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


def is_unambiguously_named(cnr: CometNameRegex) -> bool:
    return count_extracted_names(cnr) == 1


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


def count_unambiguously_named_comets(cnrs: List[CometNameRegex]) -> int:
    return sum([1 if is_unambiguously_named(cnr) else 0 for cnr in cnrs])


def count_long_period_comets(cnrs: List[CometNameRegex]) -> int:
    return sum([1 if is_long_period(cnr) else 0 for cnr in cnrs])


def count_short_period_comets(cnrs: List[CometNameRegex]) -> int:
    return sum([1 if is_short_period(cnr) else 0 for cnr in cnrs])


def count_interstellar_comets(cnrs: List[CometNameRegex]) -> int:
    return sum([1 if is_interstellar(cnr) else 0 for cnr in cnrs])


def attempt_manual_fix(comet_name: str):
    name_fixes = {
        "AsteroidP/2006VW139": "288P",
        "CometCatalinaOrbit1": "C2013US10",
        "CometCatalinaOrbit2": "C2013US10",
        "CometCatalinaOrbit3": "C2013US10",
        "CometCatalinaOrbit4": "C2013US10",
        "C/2013USCatalina": "C2013US10",
        "C/McNaught": "C/2006P1",
        "CometC/2012(ISON)": "C/2012S1",
        "CometC1/2013SidingSprings": "C/2013A1",
        "CometC2012S1(ISON)": "C/2012S1",
        "CometGarradd": "C/2009P1",
        "CometHolmes": "17P",
        "CometLulin": "C2007N3",
        "CometVL62": "C/2015VL62",
    }
    return name_fixes.get(comet_name, "Unnamed")


def decide_comet_name(cnr: CometNameRegex) -> CometRenaming:
    if is_unnamed(cnr):
        return CometRenaming(
            original_name=cnr.original_name,
            new_name=attempt_manual_fix(cnr.original_name),
        )
    elif has_naming_conflict(cnr):
        if cnr.long_period_name:
            # the long period name is very restrictive to match, so if one is found,
            # ignore the other matches as false positives
            return CometRenaming(
                original_name=cnr.original_name, new_name=str(cnr.long_period_name)
            )
        else:
            # otherwise, the new name will just show the conflict for resolution later
            return CometRenaming(
                original_name=cnr.original_name,
                new_name=f"{cnr.original_name}"
                + " ====> Long: "
                + str(cnr.long_period_name)
                + " ====> Short: "
                + str(cnr.short_period_name)
                + " ====> Interstellar: "
                + str(cnr.interstellar_name),
            )
    else:
        return CometRenaming(
            original_name=cnr.original_name,
            new_name=(cnr.long_period_name if cnr.long_period_name else "")
            + (cnr.short_period_name if cnr.short_period_name else "")
            + (cnr.interstellar_name if cnr.interstellar_name else ""),
        )


def print_conflicts(cnrs: List[CometNameRegex]) -> None:
    conflicts = [
        decide_comet_name(cnr).new_name for cnr in cnrs if has_naming_conflict(cnr)
    ]
    for conflict in conflicts:
        print(conflict)


def print_unnamed(cnrs: List[CometNameRegex]) -> None:
    unnamed = [decide_comet_name(cnr).new_name for cnr in cnrs if is_unnamed(cnr)]
    for x in unnamed:
        print(x)


def print_summary(cnrs: List[CometNameRegex]) -> None:
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

    rename_long_period_comets = [
        decide_comet_name(cnr).new_name
        for cnr in cnrs
        if (is_long_period(cnr) or has_naming_conflict(cnr))
    ]
    renamed_long_period_set = set(rename_long_period_comets)

    short_period_comets = [
        decide_comet_name(cnr).new_name for cnr in cnrs if is_short_period(cnr)
    ]
    short_period_set = set(short_period_comets)

    interstellar_comets = [
        decide_comet_name(cnr).new_name for cnr in cnrs if is_interstellar(cnr)
    ]
    interstellar_set = set(interstellar_comets)

    unnamed_comets = [decide_comet_name(cnr) for cnr in cnrs if is_unnamed(cnr)]
    manually_renamed_set = set(
        [
            (x.original_name, x.new_name)
            for x in unnamed_comets
            if x.new_name != "Unnamed"
        ]
    )
    not_manually_renamed = [
        x.original_name for x in unnamed_comets if x.new_name == "Unnamed"
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

    print(f"Manually renamed entries: ({len(manually_renamed_set)})")
    for x in sorted(manually_renamed_set, key=lambda x: x[0]):
        print(f"\t{x[0]:>30} ====> {x[1]}")
    print("")

    print(f"Entries not renamed: ({len(not_manually_renamed)})")
    for x in sorted(set(not_manually_renamed)):
        print(f"\t{x} ({not_manually_renamed.count(x)})")
    print("")

    print(
        f"Total renaming coverage: {comet_count - len(not_manually_renamed)}/{comet_count} ({100*(comet_count - len(not_manually_renamed))/comet_count}%)"
    )
    print("")


def main():
    # get list of approximate comet names from column 1
    with open("swift_comet_database.txt", "r") as f:
        # ignore first two lines
        lines = f.readlines()[2:]
        raw_comet_names = [line.split()[0] for line in lines]
        # obsids = [line.split()[1] for line in lines]

    # try to match the observation name to a conventional comet name
    comet_name_regexes = list(map(extract_comet_name, raw_comet_names))

    print_summary(comet_name_regexes)

    # produce a list of (observation name, new name)
    new_comet_names = list(map(decide_comet_name, comet_name_regexes))

    print("Random selection of 10 renamings:")
    for ncn in np.random.choice(np.array(new_comet_names), size=10):
        print(f"{ncn.original_name:>40} ===> {ncn.new_name}")


if __name__ == "__main__":
    sys.exit(main())
