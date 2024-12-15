from absl import flags
from absl import app
import os
import random
from pytorch_lightning import seed_everything

from utils import load_json_file, save_json_file

_TRAIN_DATASET_PATH = flags.DEFINE_string(
    'train_dataset_path',
    '<YOUR_PATH>',
    ''
)

_TEST_DATASET_PATH = flags.DEFINE_string(
    'test_dataset_path',
    '<YOUR_PATH>',
    ''
)

_MEMBER_NUMS = flags.DEFINE_integer(
    'member_nums',
    1000,
    ''
)

_NONMEMBER_NUMS = flags.DEFINE_integer(
    'nonmember_nums',
    1000,
    ''
)

_SAVE_DIR = flags.DEFINE_string(
    'save_dir',
    '<YOUR_DIR>',
    ''
)

_PROMPT_LENGTH = flags.DEFINE_integer(
    'prompt_length',
    128,
    ''
)

_SEED = flags.DEFINE_integer(
    'seed', 2024, '')

def reconstruct(dataset, prompt_len):
    all_dict = []
    for x in dataset:
        temp_dict = {
            'id':x['id'],
            'input': x['input'][:prompt_len]
        }
        all_dict.append(temp_dict)
    return all_dict

def main(_):
    seed_everything(_SEED.value)
    train_set = reconstruct(load_json_file(_TRAIN_DATASET_PATH.value), _PROMPT_LENGTH.value)
    test_set = reconstruct(load_json_file(_TEST_DATASET_PATH.value), _PROMPT_LENGTH.value)
    random.shuffle(train_set)
    random.shuffle(test_set)
    
    
    member_nums, nonmember_nums = _MEMBER_NUMS.value, _NONMEMBER_NUMS.value
    member_selected = train_set[ :member_nums]
    nonmember_selected = test_set[ :nonmember_nums]
    
    ''' set exist_ok=False to avoid overwriting '''
    os.makedirs(_SAVE_DIR.value, exist_ok=False)

    member_save_path = _SAVE_DIR.value + '/member' + str(member_nums) + '.json'
    nonmember_save_path = _SAVE_DIR.value + '/nonmember' + str(nonmember_nums) + '.json'
    
    save_json_file(member_save_path, member_selected)
    save_json_file(nonmember_save_path, nonmember_selected)
    
if __name__ == '__main__':
    app.run(main)