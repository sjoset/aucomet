#!/usr/bin/env python3

import sys
import pathlib
import pandas as pd


def read_swift_database_txt(fname: pathlib.Path) -> pd.DataFrame:
    """
    The original text file had fields that were not delimited by any special character,
    so raw comet names with spaces have to be handled carefully by removing the obsid from the end of the line
    and taking the rest of the line as the comet's name
    Sample rows from the file to illustrate:

    C/2019Q4Borisov                                                                      00011798001
    Comet Holmes                                                                         00030990001

    """
    raw_comet_names = []
    # get list of approximate comet names from column 1
    with open(fname, "r") as f:
        # ignore first two lines
        lines = f.readlines()[2:]

        for i, line in enumerate(lines):
            words = line.split()
            if len(words) != 2:
                print(f"Problem with line {i}: {line}")
                raw_comet_names.append(line[:-12].strip())
                print(f'Changing name to "{line[:-12].strip()}"')
            else:
                raw_comet_names.append(line.split()[0])

        obsids = [line.split()[-1] for line in lines]

    df = pd.DataFrame(
        list(zip(raw_comet_names, obsids)),
        columns=["original_swift_name", "obsid"],
    )
    return df


def write_swift_database_csv(
    fname: pathlib.Path, comet_dataframe: pd.DataFrame
) -> None:
    comet_dataframe.to_csv(fname, index=False)


def main():
    # read our data
    comet_dataframe = read_swift_database_txt(pathlib.Path("swift_comet_database.txt"))

    # write out the results
    write_swift_database_csv(
        fname=pathlib.Path("swift_comet_database.csv"), comet_dataframe=comet_dataframe
    )


if __name__ == "__main__":
    sys.exit(main())
