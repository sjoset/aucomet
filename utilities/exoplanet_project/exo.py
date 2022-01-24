#!/usr/bin/env python3

import time
import sys
import urllib
import csv
import datetime
from bs4 import BeautifulSoup
# conda install beautifulsoup4

# Builds a csv file of observable exoplanet transits on the
# given day in Auburn, AL


def utc_string_to_local_string(utcstr):
    hr, min = map(int, utcstr.split(':'))
    hr += -6
    if hr < 0:
        hr += 24
    return f"{hr:02d}:{min:02d}"


# Auburn's GMT offset is -6, function untested for other zones
# def utc_string_to_local_string(utcstr, gmtoffset):
#     hr, min = map(int, utcstr.split(':'))
#     hr += gmtoffset
#     if hr < 0:
#         hr += 24
#     return f"{hr:02d}:{min:02d}"


def build_url_list(num_days):

    url_base = 'http://var2.astro.cz/ETD/predictions.php?JDmidnight='
    url_end = '&delka=-85.48&sirka=32.7'

    # 1721424.5 is standard conversion adjustment for julian date, an extra +1 because we want
    # to advance to midnight tonight, which is technically tomorrow
    jdmid_today = datetime.datetime.today().toordinal() + 1721424.5 + 1

    url_list = [f"{url_base}{jdmid_today+i:7.05f}{url_end}" for i in range(num_days)]
    return url_list


def get_etd_targets(soup):

    # added to the beginning of urls for prediction images
    prediction_urlbase = 'http://var2.astro.cz/ETD/'

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
    print(f"Converting {target_start_utc} ...")
    target_start_local = map(utc_string_to_local_string, target_start_utc)

    # transit end times, utc and local
    # end time is inside the fourth td element in the row
    target_end_utc = [x.find_all("td")[3].contents[0] for x in good_targets]
    target_end_local = map(utc_string_to_local_string, target_end_utc)

    # magnitude of targets and the magnitude depth of transit
    # sixth td element of the row
    target_mags = [x.find_all("td")[5].contents[0] for x in good_targets]
    # seventh td element of the row
    target_mag_depths = [x.find_all("td")[6].contents[0] for x in good_targets]

    # build table
    target_data = list(zip(target_names, target_start_utc, target_end_utc, target_start_local, target_end_local, target_mags, target_mag_depths, target_RAdecs, target_preds))
    # empty row padding
    target_data.append(("", "", "", "", "", "", "", "", ""))
    return target_data


def main():

    number_of_days = 5
    output_filename = 'exoplanet_targets.csv'
    urllist = build_url_list(number_of_days)

    print(f"Downloading data for {number_of_days} days total (including today) ...")
    print("Trying these urls:")
    for i in urllist:
        print(i)
    print("")

    # all exoplanet transit data for the next couple of days
    days_from_now = 0
    for url in urllist:
        print("Downloading exoplanet data from ", url, " ...")
        page = urllib.request.urlopen(url)

        print("Done, processing ...")
        soup = BeautifulSoup(page, features="lxml")
        target_data = get_etd_targets(soup)

        print("Done, appending data to ", output_filename, " ...")
        with open(output_filename, 'a') as outfile:
            target_data_day = datetime.datetime.today() + datetime.timedelta(days=days_from_now)
            outfile.write(f"# {target_data_day.strftime('%Y-%m-%d')},,,,,,,,\n")

            csv_out = csv.writer(outfile)
            csv_out.writerow(['Name', 'Start UTC', 'End UTC', 'Start Local', 'End Local', 'Mag', 'Depth', 'Coords', 'Sky Image'])
            for row in target_data:
                csv_out.writerow(row)

        days_from_now += 1
        time.sleep(1)

    print("Done!")


if __name__ == "__main__":
    sys.exit(main())
