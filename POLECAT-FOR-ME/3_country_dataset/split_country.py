import os
import pandas as pd

# all_data = all_dataset[['Actor Name','Event Type','Event Mode','Recipient Name','Event Date','Event ID','Actor Country','Recipient Country','Country']]
def dividedByCountry(dataset,country):
    dataset_country = dataset[(dataset['Actor Country'].str.lower().str.contains(country,case=False,regex=True))|
                        (dataset['Recipient Country'].str.lower().str.contains(country,case=False,regex=True))|
                        (dataset['Country'].str.lower().str.contains(country,case=False,regex=True))]
    return dataset_country

def save_country_data(data,save_path, dataset_country):
    data = data[['Event ID', 'Subject','Relation','Object','Date', "Contexts", "ce_id", "Md5"]]
    
    #data.sort_values('Date',ascending=True,inplace=True)
    
    data_folder = os.path.join(save_path, dataset_country)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    data.to_csv(os.path.join(data_folder,'all_data.csv'),index=False)
    
    return data

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
    
save_path = "./multiagent/country_max100_min20"

all_dataset = pd.read_csv('/ThinkTank-ME/POLECAT-FOR-ME/2_historical_event/2_3_data_cleaning/data_mideast_max100_min20/MidEast_Raw.csv', keep_default_na=False)

all_country = get_country_dict()

for country1, country2 in all_country.items():
    country_i = f"{country1}|{country2}"

    # country ISO https://zh.wikipedia.org/wiki/ISO_3166-1
    all_data_i = dividedByCountry(all_dataset, country_i)
    # all_data_IRN = dividedByCountry(all_dataset,'iran|irn')
    # all_data_EGY = dividedByCountry(all_dataset,'egypt|egy')
    # all_data_SAU = dividedByCountry(all_dataset,'saudi arabia|sau')

    print(f"all_data_{country2}.shape:{all_data_i.shape}")
    # print(f"all_data_IR.shape:{all_data_IRN.shape}")
    # print(f"all_data_EG.shape:{all_data_EGY.shape}")
    # print(f"all_data_EG.shape:{all_data_SAU.shape}")

    all_data_i = save_country_data(all_data_i, save_path, f"{country1}_{country2}")
    # all_data_IRN = save_country_data(all_data_IRN, save_path, "IRN")
    # all_data_EGY = save_country_data(all_data_EGY, save_path, "EGY")
    # all_data_SAU = save_country_data(all_data_SAU, save_path, "SAU")