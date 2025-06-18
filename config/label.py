import csv

input_file = 'dataset/all_image_urls.csv'
output_file = 'dataset/labeled_dataset.csv'

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['image_url', 'category'])  # add headers

    for line in infile:
        urls = [url.strip() for url in line.strip().split(',')]
        for url in urls:
            writer.writerow([url, 'uncategorized'])  # label to update later
