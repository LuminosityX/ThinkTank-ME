import pandas as pd
import json
import os
from tqdm import tqdm


with open("../context_mideast_max100_min20.json", mode="r", encoding='latin-1') as f:
    context_json = json.load(f)

print(len(context_json))

# When testing, you need to change the name of the country in the test set to the country where the model expert belongs to, so you need to build the prompt again separately at the time of testing
def get_system_prompt(country):
    system_prompt = f"You are a think tank expert specializing in the political, military, and economic dynamics of {country}. You possess extensive knowledge of historical and contemporary events, policies, and strategies relevant to {country} and its interactions with other nations. Your expertise includes identifying patterns and trends in historical and current events, interpreting complex geopolitical contexts, and delivering informed predictions.\n\nYour task is to analyze [Historical Events] and predict the most likely object for a [Current Event to Predict]. [Historical Events] consist of a series of timestamped event sets, with each set containing multiple atomic events. The [Current Event to Predict] is presented as a query requiring the identification of the most plausible object, based on the given subject, relationship, and timestamp. Each timestamp is represented as [year-month-day], and atomic events are formatted as triples [subject, relation, object]. For example, [France, REQUEST_meet, Russia] indicates that  'France' (subject) has the relation 'REQUEST_meet' with 'Russia' (object)."
    return system_prompt

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
for filename1, filename2 in country_dict.items():
    print(f"{filename1}_{filename2}:")
    file_path_1 = os.path.join("./country_max100_min20", f"{filename1}_{filename2}")
    file_path_2 = os.path.join(file_path_1, 'all_data.csv')

    df_raw = pd.read_csv(file_path_2, sep=',',  keep_default_na=False)
    real_df = df_raw[df_raw['ce_id']!=-1]

    train_real_df = real_df[real_df["Date"]<"2024-01-01"]
    test_real_df = real_df[real_df["Date"]>="2024-01-01"]

    train_datasets_list = []
    train_datasets = {}
    val_datasets = {}

    for _, event_i in train_real_df.iterrows():
        s_i = event_i["Subject"]
        r_i = event_i["Relation"]
        o_i = event_i["Object"]
        date_i = event_i["Date"]
        ce_id_i = str(event_i["ce_id"])

        context_i = context_json[ce_id_i][date_i]

        if context_i != "":
            country_name_i = country_name_dict[filename1]
            system_i = get_system_prompt(country_name_i)
            current_event =f"Event on {date_i} = [{s_i}, {r_i}, ?]"
            target = o_i

            input_i = f"[Historical Events]:\n{context_i}\n[Current Event to Predict]:\n{current_event}\n\nPlease use your deep understanding of {country_name_i}'s internal and external dynamics, as well as the broader geopolitical environment, to deliver accurate, contextually informed predictions. Only provide the missing object without additional explanation."

            # print(f"System prompt:\n{system_i}\n\nUser:\n{input_i}\n\nAssistant:\n{target}")

            train_datasets_list.append({"system_prompt": system_i, "input": input_i, "target": target, "ce_id": ce_id_i})
        
    for index, event_i in enumerate(train_datasets_list):
        train_datasets[index] = event_i

    # Select the last 1000 examples from train_datasets_list for validation
    val_datasets_list = train_datasets_list[-1000:] if len(train_datasets_list) >= 1000 else train_datasets_list[:]
    for index, event_i in enumerate(val_datasets_list):
        val_datasets[index] = event_i

    with open(f"{file_path_1}/val_object.json", mode="w", encoding='latin-1') as f:
        json.dump(val_datasets, f, indent=4)

    print(f"{filename1} train: {len(train_datasets)} {len(train_real_df)}(with CE first day event)")

    with open(f"{file_path_1}/train_object.json", mode="w", encoding='latin-1') as f:
        json.dump(train_datasets, f, indent=4)

    test_datasets_list = []
    test_datasets = {}

    for _, event_i in test_real_df.iterrows():
        s_i = event_i["Subject"]
        r_i = event_i["Relation"]
        o_i = event_i["Object"]
        date_i = event_i["Date"]
        ce_id_i = str(event_i["ce_id"])

        context_i = context_json[ce_id_i][date_i]

        if context_i != "":
            country_name_i = country_name_dict[filename1]
            system_i = get_system_prompt(country_name_i)
            current_event =f"Event on {date_i} = [{s_i}, {r_i}, ?]"
            target = o_i

            input_i = f"[Historical Events]:\n{context_i}\n[Current Event to Predict]:\n{current_event}\n\nPlease use your deep understanding of {country_name_i}'s internal and external dynamics, as well as the broader geopolitical environment, to deliver accurate, contextually informed predictions. Only provide the missing object without additional explanation."

            test_datasets_list.append({"system_prompt": system_i, "input": input_i, "target": target, "ce_id": ce_id_i})
        
    for index, event_i in enumerate(test_datasets_list):
        test_datasets[index] = event_i

    print(f"{filename1} test: {len(test_datasets)} {len(test_real_df)}(with CE first day event)")

    with open(f"{file_path_1}/test_object.json", mode="w", encoding='latin-1') as f:
        json.dump(test_datasets, f, indent=4)

    print()