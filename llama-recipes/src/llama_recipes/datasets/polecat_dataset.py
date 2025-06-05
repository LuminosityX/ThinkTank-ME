import json
import torch
from torch.utils.data import Dataset

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

class polecat(Dataset):
    def __init__(
        self,
        dataset_config,
        tokenizer,
        split,
    ):
        self.config = dataset_config
        self.tokenizer = tokenizer
        with open(split, mode="r", encoding='latin-1') as file:
            self.training_file = json.load(file)
        self.country_name_dict = get_country_name()

        split_last = split.split("/")[-1]
        
        if "test" in split_last:
            self.config.is_test = True
            print(f"Due the split ({split_last}) has the test string, so set the is_test True")

    def convert_to_features(self, example_batch):

        # Create prompt and tokenize contexts and questions

        system_ = example_batch["system_prompt"]
        user_ = example_batch["input"]
        target_ = example_batch["target"]

        # for Non-Training and All-Training
        if self.config.general_system_prompt:
            system_ = f"You are a think tank expert specializing in the political, military, and economic dynamics. Your expertise includes identifying patterns and trends in historical and current events, interpreting complex geopolitical contexts, and delivering informed predictions.\n\nYour task is to analyze [Historical Events] and predict the most likely object for a [Current Event to Predict]. [Historical Events] consist of a series of timestamped event sets, with each set containing multiple atomic events. The [Current Event to Predict] is presented as a query requiring the identification of the most plausible object, based on the given subject, relationship, and timestamp. Each timestamp is represented as [year-month-day], and atomic events are formatted as triples [subject, relation, object]. For example, [France, REQUEST_meet, Russia] indicates that  'France' (subject) has the relation 'REQUEST_meet' with 'Russia' (object)."
            split_user = user_.split("\n\nPlease ")
            assert len(split_user) == 2
            assert split_user[-1][:3] == "use"

            user_ = split_user[0] + f"\n\nPlease use your deep understanding of the broader geopolitical environment, to deliver accurate, contextually informed predictions. Only provide the missing object without additional explanation."
        
        # test should use model expert name
        elif self.config.is_test and self.config.test_model_name:
            COUNTRY_NAME = self.config.test_model_name
            assert COUNTRY_NAME != ""

            if '_' in COUNTRY_NAME:
                COUNTRY_NAME = COUNTRY_NAME.replace("_", " ")
            country_name = self.country_name_dict[COUNTRY_NAME]
            system_ = f"You are a think tank expert specializing in the political, military, and economic dynamics of {country_name}. You possess extensive knowledge of historical and contemporary events, policies, and strategies relevant to {country_name} and its interactions with other nations. Your expertise includes identifying patterns and trends in historical and current events, interpreting complex geopolitical contexts, and delivering informed predictions.\n\nYour task is to analyze [Historical Events] and predict the most likely object for a [Current Event to Predict]. [Historical Events] consist of a series of timestamped event sets, with each set containing multiple atomic events. The [Current Event to Predict] is presented as a query requiring the identification of the most plausible object, based on the given subject, relationship, and timestamp. Each timestamp is represented as [year-month-day], and atomic events are formatted as triples [subject, relation, object]. For example, [France, REQUEST_meet, Russia] indicates that  'France' (subject) has the relation 'REQUEST_meet' with 'Russia' (object)."
            split_user = user_.split("\n\nPlease ")
            assert len(split_user) == 2
            assert split_user[-1][:3] == "use"

            user_ = split_user[0] + f"\n\nPlease use your deep understanding of {country_name}'s internal and external dynamics, as well as the broader geopolitical environment, to deliver accurate, contextually informed predictions. Only provide the missing object without additional explanation."


        messages = [
            {"role": "system", "content": system_},
            {"role": "user", "content": user_},
            {"role": "assistant", "content": target_}
        ]

        training_example = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False)

        last_index = None
        for i, num in reversed(list(enumerate(training_example))):
            if num == 128007:
                last_index = i
                break

        input_ids = training_example[:last_index+1]
        label_ids = training_example[last_index+1:]

        if self.config.is_test:
            sample = {
                "input_ids": input_ids + label_ids,
                "attention_mask": [1] * len(input_ids + label_ids),
                "labels": [-100] * len(input_ids) + label_ids,
                "prompt_ids": input_ids,
                "prompt_attention_mask": [1] * len(input_ids),
                "labels_ids": label_ids
            }
        else:
            sample = {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": [-100] * len(input_ids) + label_ids,
            }
        return sample

    def __getitem__(self, index):
        key_index = str(index)
        if key_index not in self.training_file:
            raise IndexError(f"Index {index} out of range")
        return self.convert_to_features(self.training_file[key_index])
    
    def __len__(self):
        return len(self.training_file)

def get_preprocessed_polecat(dataset_config, tokenizer, split: str):
    dataset = polecat(
        dataset_config=dataset_config,
        tokenizer=tokenizer,
        split=split,
    )
    return dataset
