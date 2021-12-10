#!/usr/bin/env python3

import sys
import urllib
import csv
from bs4 import BeautifulSoup
# conda install beautifulsoup4

# Builds a csv file of observable exoplanet transits on the
# given day in Auburn, AL


def main():

    # coords for Auburn
    baseurl = 'http://var2.astro.cz/ETD/predictions.php?delka=-85.48&sirka=32.7'
    # added to the beginning of urls for prediction images
    prediction_urlbase = 'http://var2.astro.cz/ETD/'
    output_filename = 'exoplanet_targets.csv'

    print("Downloading exoplanet data from ", baseurl, " ...")
    page = urllib.request.urlopen(baseurl)
    print("Done, processing ...")
    soup = BeautifulSoup(page, features="lxml")

    # 2 other tables before the exoplanet table
    exoplanet_table = soup.find_all("table")[2]

    # first 2 table rows are column names and spacer
    targets = exoplanet_table.find_all("tr")[2:]

    # take every other row to filter out spacing rows
    good_targets = targets[0::2]

    # grab the names
    target_names = [p.find("a").contents[0] for p in good_targets]

    # get RA
    # RA is third entry in the span tag, strip the 'RA: ' part at beginning of string
    target_RAs = [x.find("span").contents[2][4:] for x in good_targets]

    # get dec
    # dec is fifth entry in the span tag, strip the 'DE: ' part at beginning of string
    target_decs = [x.find("span").contents[4][4:] for x in good_targets]

    # combine them for easy entry for web tools
    target_RAdecs = [x + ' ' + y for (x, y) in zip(target_RAs, target_decs)]

    # get sky image prediction links
    target_preds = [prediction_urlbase + x.find("a")['href'] for x in good_targets]

    # transit start times, utc and local
    # start time is inside the second td element in the row
    target_start_utc = [x.find_all("td")[1].contents[0] for x in good_targets]
    target_start_local = []
    for x in target_start_utc:
        hr, min = map(int, x.split(':'))
        hr -= 6
        if hr < 0:
            hr += 24
        target_start_local.append(f"{hr:02d}:{min:02d}")

    # transit end times, utc and local
    # end time is inside the fourth td element in the row
    target_end_utc = [x.find_all("td")[3].contents[0] for x in good_targets]
    target_end_local = []
    for x in target_end_utc:
        hr, min = map(int, x.split(':'))
        hr -= 6
        if hr < 0:
            hr += 24
        target_end_local.append(f"{hr:02d}:{min:02d}")

    # magnitude of targets and the magnitude depth of transit
    # sixth td element of the row
    target_mags = [x.find_all("td")[5].contents[0] for x in good_targets]
    # seventh td element of the row
    target_mag_depths = [x.find_all("td")[6].contents[0] for x in good_targets]

    # build table
    target_data = list(zip(target_names, target_start_utc, target_end_utc, target_start_local, target_end_local, target_mags, target_mag_depths, target_RAdecs, target_preds))

    print("Done, writing data to ", output_filename, " ...")

    # write table
    with open(output_filename, 'w') as outfile:
        csv_out = csv.writer(outfile)
        csv_out.writerow(['Name', 'Start UTC', 'End UTC', 'Start Local', 'End Local', 'Mag', 'Depth', 'Coords', 'Sky Image'])
        for row in target_data:
            csv_out.writerow(row)

    print("Done!")


if __name__ == "__main__":
    sys.exit(main())
