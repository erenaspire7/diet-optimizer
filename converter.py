import json

import csv


def array_to_csv(array, filename):
    # Write array to CSV file
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        # For 1D array, we need to wrap each element in a list to write as a row
        for item in array:
            writer.writerow([item])


# PATH = "results/de/v2/de-diet-optimizer-fin.json"
# PATH = "results/ga/ga-diet-optimizer-v10.json"
# PATH = "de-diet-optimizer-v1.json"
PATH = "ga-diet-optimizer-v1.json"


with open(PATH, "r") as file:
    data = json.load(file)

    my_array = [el["weight"] for el in data["best_individual"]["params"]]

    array_to_csv(my_array, "output.csv")
