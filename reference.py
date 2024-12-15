from absl import app
from absl import flags
import os
import torch

from utils import load_json_file, save_json_file

_LOSS_FILE_DIR = flags.DEFINE_string(
    'loss_file_dir',
    '<YOUR_DIR>',
    ''
)

_REFERENCE_NAME = flags.DEFINE_string(
    'reference_name',
    '<NAME>',
    ''
)

_TARGET_NAME = flags.DEFINE_string(
    'target_name',
    '<NAME>',
    ''
)


_SOFTMAX_SAVE_DIR = flags.DEFINE_string(
    'softmax_save_dir',
    '<YOUR_DIR>',
    ''
)


def main(_):
    print("------------------ load loss data -----------------")
    reference_file_path = _LOSS_FILE_DIR.value + _REFERENCE_NAME.value + '.json'
    target_file_path = _LOSS_FILE_DIR.value + _TARGET_NAME.value + '.json'
    reference_ll_list = load_json_file(reference_file_path)
    target_ll_list = load_json_file(target_file_path)
    
    
    print("------------------ loss difference -----------------")
    member_reference_ll_list = [dic for dic in reference_ll_list if dic['type']=='member']
    nonmember_reference_ll_list = [dic for dic in reference_ll_list if dic['type']=='nonmember']
    member_target_ll_list = [dic for dic in target_ll_list if dic['type']=='member']
    nonmember_target_ll_list = [dic for dic in target_ll_list if dic['type']=='nonmember']
    
    member_diff_list = [member_target_ll_list[i]['loss'] - member_reference_ll_list[i]['loss'] for i in range(len(member_target_ll_list))]
    nonmember_diff_list = [nonmember_target_ll_list[i]['loss'] - nonmember_reference_ll_list[i]['loss'] for i in range(len(nonmember_target_ll_list))]
    
    print("------------------ softmax rank -----------------")
    member_diff_id_list = [{'type': 'member', 'id':i, 'loss_diff': round(item, 3)} for i,item in enumerate(member_diff_list)]
    nonmember_diff_id_list = [{'type': 'nonmember', 'id':i+len(member_diff_id_list), 'loss_diff': round(item, 3)} for i,item in enumerate(nonmember_diff_list)]
    diff_id_list = member_diff_id_list + nonmember_diff_id_list
    
    diff_list = member_diff_list + nonmember_diff_list
    ll_softmax = torch.nn.functional.softmax(torch.tensor(diff_list), dim=-1)
    ll_softmax = [round(float(x), 4) for x in ll_softmax]
    
    for dic in diff_id_list:
        dic['loss_softmax'] = ll_softmax[dic['id']]
    
    os.makedirs(_SOFTMAX_SAVE_DIR.value, exist_ok=True)
    save_path = _SOFTMAX_SAVE_DIR.value + '/' +  _TARGET_NAME.value + '-' + _REFERENCE_NAME.value + '.json'
    save_json_file(save_path, diff_id_list)


if __name__ == '__main__':
    app.run(main)