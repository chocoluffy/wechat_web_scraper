# create a csv file that contains no content. copy from 'wordvec-en-vec160904.csv'
import csv

base_dir = './data/'
wordvec_file = base_dir + 'wordvec-en-vec160904.csv'
new_file = base_dir + 'new_wordvec-en-vec160904.csv'
ada_content_title = base_dir + 'wordvec160904.csv'


title_lst = []
with open(ada_content_title, 'r') as source:
    reader = csv.DictReader(source) # A list of all rows, with posts[-1] the most recent one.
    for row in reader:
    	title_lst.append(row['title'])


with open(new_file, 'w') as target:
    fieldnames = ['id', 'title', 'literal_title', 'tags', 'wordvec', 'url']
    writer = csv.DictWriter(target, fieldnames=fieldnames)
    writer.writeheader()
    with open(wordvec_file, 'r') as source:
	    reader = csv.DictReader(source) # A list of all rows, with posts[-1] the most recent one.
	    for row in reader:
	    	rowid = int(row['id'])
	    	newrow = {'id': row['id'], 'title': title_lst[rowid], 'literal_title': title_lst[rowid], 'tags': row['tags'], 'wordvec': row['wordvec'], 'url': row['url']}
	        writer.writerow(newrow)