#!/usr/bin/env python3

import sys
import re
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class CometNameRegex:
    original_name: str
    long_period_name: Optional[str]
    short_period_name: Optional[str]
    interstellar_name: Optional[str]


def extract_comet_name(comet_name: str) -> CometNameRegex:
    """
    Takes a string, searches for comet naming formats, and extracts it into the CometNameRegex dataclass
    """
    # match naming convention of long period comets: C/[YEAR][one optional space character][one to two letters][one to two digits]
    long_period_match = re.search("C/[0-9]{4}\\s?[A-Z]{1,2}[0-9]{1,2}", comet_name)

    # match naming convention of short period comets: [1 to 3 digits]P
    short_period_match = re.search("[0-9]{1,3}P", comet_name)

    # try interstellar interstellar naming scheme
    interstellar_match = re.search("[0-9]{1,3}I/", comet_name)

    # get the matched part of the string if there was a match
    long_period_name = long_period_match.group() if long_period_match else None
    short_period_name = short_period_match.group() if short_period_match else None
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


def decide_comet_name(cnr: CometNameRegex) -> str:
    if is_unnamed(cnr):
        # here we would define a dictionary to translate bad entries to their actual names
        # return cnr.original_name + "_FIXME"
        return cnr.original_name
    elif has_naming_conflict(cnr):
        # the long period name is very restrictive to match, so if one is found,
        # ignore the other matches as false positives
        if cnr.long_period_name:
            return str(cnr.long_period_name)
        else:
            return (
                f"{cnr.original_name}"
                + " ====> Long: "
                + str(cnr.long_period_name)
                + " ====> Short: "
                + str(cnr.short_period_name)
                + " ====> Interstellar: "
                + str(cnr.interstellar_name)
            )
    else:
        return (
            # cnr.original_name
            # + " ====> "
            (cnr.long_period_name if cnr.long_period_name else "")
            + (cnr.short_period_name if cnr.short_period_name else "")
            + (cnr.interstellar_name if cnr.interstellar_name else "")
        )


def print_conflicts(cnrs: List[CometNameRegex]) -> None:
    conflicts = [decide_comet_name(cnr) for cnr in cnrs if has_naming_conflict(cnr)]
    for conflict in conflicts:
        print(conflict)


def print_unnamed(cnrs: List[CometNameRegex]) -> None:
    unnamed = [decide_comet_name(cnr) for cnr in cnrs if is_unnamed(cnr)]
    for x in unnamed:
        print(x)


def print_summary(cnrs: List[CometNameRegex]) -> None:
    comet_count = len(cnrs)
    confict_count = count_comet_name_conflicts(cnrs)
    unnamed_count = count_unnamed_comets(cnrs)
    success_count = comet_count - unnamed_count
    print(f"Unambiguously named: {comet_count - confict_count - unnamed_count}")
    print(f"Conflicts: {confict_count}")
    print(f"Unnamed: {unnamed_count}")
    print(
        f"Success assuming conflicts are false positives: {success_count} out of {comet_count} ({100 * (success_count)/comet_count:4.1f} %)"
    )
    print("")

    long_period_count = count_long_period_comets(cnrs)
    short_period_count = count_short_period_comets(cnrs)
    interstellar_count = count_interstellar_comets(cnrs)
    print("Comet observation count summary:")
    print(f"\tLong period: {long_period_count}")
    print(f"\tShort period: {short_period_count}")
    print(f"\tinterstellar: {interstellar_count}")
    print("")

    regularized_comet_names = list(map(decide_comet_name, cnrs))
    comet_name_set = set(regularized_comet_names)

    long_period_comets = [
        decide_comet_name(cnr)
        for cnr in cnrs
        if (is_long_period(cnr) or has_naming_conflict(cnr))
    ]
    long_period_set = set(long_period_comets)

    short_period_comets = [
        decide_comet_name(cnr) for cnr in cnrs if is_short_period(cnr)
    ]
    short_period_set = set(short_period_comets)

    interstellar_comets = [
        decide_comet_name(cnr) for cnr in cnrs if is_interstellar(cnr)
    ]
    interstellar_set = set(interstellar_comets)

    unnamed_set = set([decide_comet_name(cnr) for cnr in cnrs if is_unnamed(cnr)])

    print("Long period comets:")
    for x in long_period_set:
        print(f"\t{x}")
    print("")

    print("Short period comets:")
    for x in short_period_set:
        print(f"\t{x}")
    print("")

    print("Unnamed entries:")
    for x in unnamed_set:
        print(f"\t{x}")
    print("")

    print("\n------------")
    print(f"Unique comet entries: {len(comet_name_set)}")
    print("------------")

    print(f"Unique long-period comets: {len(long_period_set)}")

    print(f"Unique short-period comets: {len(short_period_set)}")

    print(f"Unique interstellar comets: {len(interstellar_set)}")

    print(f"Unique unnamed: {len(unnamed_set)}")

    print(
        f"\tTotal: {len(long_period_set) + len(short_period_set) + len(interstellar_set) + len(unnamed_set)}"
    )


def main():
    # get list of approximate comet names from column 1
    with open("swift_comet_database.txt", "r") as f:
        # ignore first two lines
        lines = f.readlines()[2:]
        raw_comet_names = [line.split()[0] for line in lines]

    comet_name_regexes = list(map(extract_comet_name, raw_comet_names))
    print_summary(comet_name_regexes)


if __name__ == "__main__":
    sys.exit(main())
