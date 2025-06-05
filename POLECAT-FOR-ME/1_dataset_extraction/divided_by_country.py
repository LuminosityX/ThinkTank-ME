import csv
import os
import random
from tqdm import tqdm
#from country2regions import regions
import pandas as pd
from datetime import datetime


total_row_count = 0
unique_article = set()
count_after_filter = 0
country_is_none = 0

def csv_headers():
    return ['Event ID', 'City', 'Story Organizations', 'Country', 'Longitude', 'Province', 'Version',
            'Recipient Country', 'Latitude', 'Primary Recipient Sector', 'Story People', 'Actor Name Raw',
            'Story Locations', 'Quad Code', 'Wikipedia Actor ID', 'Raw Placename', 'Event Mode', 'Primary Actor Sector',
            'Recipient Title', 'Contexts', 'Actor COW', 'Actor Country', 'Recipient COW', 'Wikipedia Recipient ID',
            'Event Date', 'Source', 'Placename', 'Feature Type', 'GeoNames ID', 'Recipient Name', 'District',
            'Language', 'Actor Name', 'Actor Title', 'Recipient Sectors', 'Event Type', 'Publication Date',
            'Actor Sectors', 'Recipient Name Raw']

def get_country_dict():
    return {'iran': 'irn',   'israel': 'isr',               'egypt': 'egy',   'saudi arabia': 'sau',
            'turkey': 'tur', 'iraq': 'irq',                 'yemen': 'yem',   'syria': 'syr',
            'jordan': 'jor', 'united arab emirates': 'are', 'lebanon': 'lbn', 'oman': 'omn',
            'kuwait': 'kwt', 'qatar': 'qat',                'bahrain': 'bhr', 'cyprus': 'cyp',
            'palestine': 'pse', 
            'china': 'chn',     'united states': 'usa',   'russia': 'rus', 'united kingdom': 'gbr', 'france': 'fra',
            'germany': 'deu',   'korea': 'kor',     'japan': 'jpn',  'india': 'ind',     'canada': 'can', 
            'italy': 'ita',     'australia': 'aus', 'spain': 'esp',  'argentina': 'arg', 'brazil': 'bra',
            'indonesia': 'idn', 'mexico': 'mex',    'south africa': 'zaf'}

# def country_to_region(country_name):
#     for region, countries in regions.items():
#         if country_name in countries:
#             return region
#     return None


def process_csv(directory):

    ### get country list
    country_list = []
    country_dict = get_country_dict()
    for key_i, value_i in country_dict.items():
        country_list.append(key_i)
        country_list.append(value_i)
    print(f'Country List: {country_list}')
    ###

    global count_after_filter
    rows = []
    files = os.listdir(directory)
    for filename in tqdm(files, desc='Processing files'):
        file_path = os.path.join(directory, filename)
        with open(file_path, mode='r', encoding='latin-1') as file:
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                if process_row(row, country_list):
                    count_after_filter += 1
                    rows.append(row)

    
    # save to csv
    df = pd.DataFrame(rows)
    df.to_csv('./data_mideast/MidEast.csv', index=False)

def process_row(row, country_list):
    global total_row_count, unique_article, country_is_none
    total_row_count += 1
    country_fields = ['Country', 'Recipient Country', 'Actor Country']
    article_id = row['Event ID'].split('_')[0]
    unique_article.add(article_id)
    for field in country_fields:
        if row.get(field):
            # country filter
            countries = set(
                name.strip() for sublist in [part.split('|') for part in row[field].split(';')] for name in sublist)
            for country in countries:
                # keep Iran, Israel, and Egypt
                if country.lower() in country_list:
                    return True

    return False


if __name__ == "__main__":
    directory = 'PATH_TO_ORIGINAL_FILES'
    process_csv(directory)
    print(f"Total row count: {total_row_count}")
    print(f"Unique article count: {len(unique_article)}")
    print(f"Count after filter: {count_after_filter}")
