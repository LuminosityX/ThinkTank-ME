import pandas as pd
import json
import os
from tqdm import tqdm


def get_country_dict():
    return {'iran': 'irn',   'israel': 'isr',               'egypt': 'egy',   'saudi arabia': 'sau',
            'turkey': 'tur', 'iraq': 'irq',                 'yemen': 'yem',   'syria': 'syr',
            'jordan': 'jor', 'united arab emirates': 'are', 'lebanon': 'lbn', 'oman': 'omn',
            'kuwait': 'kwt', 'qatar': 'qat',                'bahrain': 'bhr', 'cyprus': 'cyp',
            'palestine': 'pse', }

def get_country_name():
    return {'iran': 'Iran',   'israel': 'Israel',               'egypt': 'Egypt',   'saudi arabia': 'Saudi Arabia',
            'turkey': 'Turkey', 'iraq': 'Iraq',                 'yemen': 'Yemen',   'syria': 'Syria',
            'jordan': 'Jordan', 'united arab emirates': 'United Arab Emirates', 'lebanon': 'Lebanon', 'oman': 'Oman',
            'kuwait': 'Kuwait', 'qatar': 'Qatar',                'bahrain': 'Bahrain', 'cyprus': 'Cyprus',
            'palestine': 'Palestine', 
            'china': 'China',     'united states': 'United States',   'russia': 'Russia', 'united kingdom': 'United Kingdom', 'france': 'France',
            'germany': 'Germany',   'korea': 'Korea',     'japan': 'Japan',  'india': 'India',     'canada': 'Canada', 
            'italy': 'Italy',     'australia': 'Australia', 'spain': 'Spain',  'argentina': 'Argentina', 'brazil': 'Brazil',
            'indonesia': 'Indonesia', 'mexico': 'Mexico',    'south africa': 'South Africa'}


country_dict = get_country_dict()
country_name_dict = get_country_name()

# all test dataset

all_country_json = {}
all_country_index = 0

for filename1, filename2 in country_dict.items():
    print(f"{filename1}_{filename2}:")
    file_path_1 = os.path.join("./country_max100_min20", f"{filename1}_{filename2}")
    file_path_2 = os.path.join(file_path_1, 'test_object.json')

    with open(file_path_2, mode="r", encoding='latin-1') as f:
        test_json_file = json.load(f)


    for _, content_i in test_json_file.items():
        all_country_json[str(all_country_index)] = {"system_prompt": content_i["system_prompt"], "input": content_i["input"], "target": content_i["target"], "ce_id": content_i["ce_id"], "country_name": filename1, "country_code": filename2, "number_country": int(_)}
        all_country_index = all_country_index + 1
    
    print(f"length: {len(test_json_file)}")
    
print(f"ALL: {len(all_country_json)}")

with open(f"./country_max100_min20/mideast_test_object.json", mode="w", encoding='latin-1') as f:
    json.dump(all_country_json, f, indent=4)

# all val dataset

all_country_json = {}
all_country_index = 0

for filename1, filename2 in country_dict.items():
    print(f"{filename1}_{filename2}:")
    file_path_1 = os.path.join("./country_max100_min20", f"{filename1}_{filename2}")
    file_path_2 = os.path.join(file_path_1, 'val_object.json')

    with open(file_path_2, mode="r", encoding='latin-1') as f:
        train_json_file = json.load(f)

    for _, content_i in train_json_file.items():
        all_country_json[str(all_country_index)] = {"system_prompt": content_i["system_prompt"], "input": content_i["input"], "target": content_i["target"], "ce_id": content_i["ce_id"], "country_name": filename1, "country_code": filename2, "number_country": int(_)}
        all_country_index = all_country_index + 1
    
    print(f"length: {len(train_json_file)}")
    
print(f"ALL: {len(all_country_json)}")

with open(f"./country_max100_min20/mideast_val_object.json", mode="w", encoding='latin-1') as f:
    json.dump(all_country_json, f, indent=4)

# all train dataset

all_country_json = {}
all_country_index = 0

for filename1, filename2 in country_dict.items():
    print(f"{filename1}_{filename2}:")
    file_path_1 = os.path.join("./country_max100_min20", f"{filename1}_{filename2}")
    file_path_2 = os.path.join(file_path_1, 'train_object.json')

    with open(file_path_2, mode="r", encoding='latin-1') as f:
        train_json_file = json.load(f)


    for _, content_i in train_json_file.items():
        all_country_json[str(all_country_index)] = {"system_prompt": content_i["system_prompt"], "input": content_i["input"], "target": content_i["target"], "ce_id": content_i["ce_id"], "country_name": filename1, "country_code": filename2, "number_country": int(_)}
        all_country_index = all_country_index + 1
    
    print(f"length: {len(train_json_file)}")
    
print(f"ALL: {len(all_country_json)}")

with open(f"./country_max100_min20/mideast_train_object.json", mode="w", encoding='latin-1') as f:
    json.dump(all_country_json, f, indent=4)