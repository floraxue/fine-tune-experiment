import csv


def read_from_csv():
    fname = '../fine-tune-data/cats_vs_dogs_submission_2018-02-04-09-25-18.csv'
    file = open(fname, 'r', newline='')
    count_correct = 0
    with file:
        reader = csv.DictReader(file)
        for row in reader:
        	if int(row['id']) <= 1000:
        		if int(row['label']) == 0:
        			count_correct += 1
        	else:
        		if int(row['label']) == 1:
        			count_correct += 1
    file.close()
    print(count_correct / 2000)

read_from_csv()
