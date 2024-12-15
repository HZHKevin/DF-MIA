from absl import flags
from absl import app
import os

from utils import load_json_file, save_json_file


_PERTURB_DATA_PATH = flags.DEFINE_string(
    'perturb_data_path',
    '<YOUR_PATH>',
    ''
)

_SOFTMAX_RANK_LIST = flags.DEFINE_string(
    'softmax_rank_path',
    '<YOUR_PATH>',
    ''
)

_TOTALL_NUMS = flags.DEFINE_integer(
    'total_nums',
    50000,
    ''
)

_NEIGHBOUR_NUMS = flags.DEFINE_integer(
    'neighbour_nums',
    60,
    ''
)

_SAVE_DIR = flags.DEFINE_string(
    'save_dir',
    '<YOUR_DIR>',
    ''
)

_SAVE_NAME_PREFIX = flags.DEFINE_string(
    'save_name_prefix',
    '<TEST>',
    ''
)



def main(_):
    neighbour_batch = load_json_file(_PERTURB_DATA_PATH.value)

    softmax_rank_list = load_json_file(_SOFTMAX_RANK_LIST.value)
    softmax_rank_list_sorted = sorted(softmax_rank_list, key=lambda x: x['loss_diff'], reverse=False)

    sft_list = []
    count = 0
    for dic in softmax_rank_list_sorted:
        selected_num = max(int(_TOTALL_NUMS.value * dic['loss_softmax']), 1)
        for i in range(selected_num):
            if count >=1000:
                temp_dict = {
                    'id': dic['id'],
                    'input': neighbour_batch[dic['id']]['input_neighbour_batch'][i % _NEIGHBOUR_NUMS.value],
                    }
            else:
                temp_dict = {
                    'id': dic['id'],
                    'input': neighbour_batch[dic['id']]['ori_input'],
                    }
            sft_list.append(temp_dict)
        count+=1
    print(len(sft_list), sft_list[0])
    
    sft_formatted_list = [dic['input'] for dic in sft_list]
    print(len(sft_formatted_list), sft_formatted_list[0])
    
    
    os.makedirs(_SAVE_DIR.value, exist_ok=True)
    
    save_path = _SAVE_DIR.value + "/" + _SAVE_NAME_PREFIX.value + "_" + str(len(sft_list)) + '.json'
    save_json_file(save_path, sft_formatted_list)

    
if __name__ == '__main__':
    app.run(main)
