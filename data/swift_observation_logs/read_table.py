#!/usr/bin/env python3

import sys
import pandas as pd


def main():
    c = pd.read_csv("./73P_log.csv")
    print(c.columns)


if __name__ == "__main__":
    sys.exit(main())
