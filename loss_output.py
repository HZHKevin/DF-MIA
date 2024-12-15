from absl import app
from absl import flags
import os

from utils import load_model_and_tokenizer, load_json_file, save_json_file, get_ll_list

_SAVE_DOT = flags.DEFINE_integer(
    'save_dot',
    2,
    ''
)

_SAVE_NAME = flags.DEFINE_string(
    'save_name',
    '<SAVE_NAME>',
    ''
)

_MODEL_PATH = flags.DEFINE_string(
    'model_path',
    '<YOUR_PATH>',
    ''
)

_USE_FLASH_ATTN = flags.DEFINE_bool(
    'use_flash_attn',
    False,
    ''
)

_MEMBER_PATH = flags.DEFINE_string(
    'member_path',
    '<YOUR_PATH>',
    ''
)

_NONMEMBER_PATH = flags.DEFINE_string(
    'nonmember_path',
    '<YOUR_PATH>',
    ''
)

_SAVE_DIR = flags.DEFINE_string(
    'save_dir',
    '<YOUR_DIR>',
    ''
)



def main(_):
    model, tokenizer = load_model_and_tokenizer(_MODEL_PATH.value, _USE_FLASH_ATTN.value)
    member_subset = [dic['input'] for dic in load_json_file(_MEMBER_PATH.value)]
    nonmember_subset = [dic['input'] for dic in load_json_file(_NONMEMBER_PATH.value)]
    member_ll_list = get_ll_list(member_subset, model, tokenizer, _SAVE_DOT.value)
    nonmember_ll_list = get_ll_list(nonmember_subset, model, tokenizer, _SAVE_DOT.value)

    member_len, nonmember_len = len(member_ll_list), len(nonmember_ll_list)
    ll_list_merge = member_ll_list + nonmember_ll_list
    all_dict = []
    for i, item in enumerate(ll_list_merge):
        temp_dict = {
            'type': ('member' if i<member_len else 'nonmember'),
            'id': i,
            'loss': item
        }
        all_dict.append(temp_dict)
    
    os.makedirs(_SAVE_DIR.value, exist_ok=True)
    save_path = _SAVE_DIR.value + "/" + _SAVE_NAME.value + ".json"
    save_json_file(save_path, all_dict)
    
if __name__ == '__main__':
    app.run(main)