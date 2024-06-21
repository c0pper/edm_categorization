import csv

import csv

def reorder_csv_results(reference_file, experiment_file):
    # Store rows from the reference CSV file in a dictionary
    reference_data = {}
    with open(reference_file, 'r', encoding="utf-8-sig") as ref_csv:
        ref_reader = csv.reader(ref_csv)
        next(ref_reader)  # Skip header row
        for row in ref_reader:
            reference_data[row[0]] = row

    # Read the experiment CSV file and reorder rows based on the reference
    experiment_rows = set()
    with open(experiment_file, 'r', encoding="utf-8-sig") as exp_csv:
        exp_reader = csv.reader(exp_csv, delimiter=";")
        next(exp_reader)
        for row in exp_reader:
            experiment_rows.add(tuple(row))
    
    reordered_rows = []
    for r in list(reference_data.items()):
        for exp_row in experiment_rows:
            title = exp_row[0].split("### DESCRIPTION:\n ")[0].replace("### TITLE:\n ", "").strip()
            desc = exp_row[0].split("### DESCRIPTION:\n ")[1].strip()
            if title + " " + desc in r[0]:
                reordered_rows.append(exp_row)

    # Write reordered rows to a new CSV file
    output_file = experiment_file[:-4] + "_reordered.csv"
    with open(output_file, 'w', newline='', encoding="utf-8") as output_csv:
        writer = csv.writer(output_csv, delimiter=";")
        writer.writerows(reordered_rows)




if __name__ == "__main__":
    reorder_csv_results('experiment_linear_svm_6a523988.csv', 'experiment_elmib_6b2091b0.csv')