from absl import flags
from absl import app
import os

from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pytorch_lightning import seed_everything

from utils import load_json_file, save_json_file, perturb_texts

_NEIGHBOUR_NUMS = flags.DEFINE_integer(
    'neighbour_nums',
    60,
    ''
)

_PERTURB_PC = flags.DEFINE_float(
    'perturb_pc',
    0.3,
    ''
)

_MASK_MODEL_DIR = flags.DEFINE_string(
    'mask_model_dir',
    '<YOUR_DIR>',
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

_ID_BEGIN = flags.DEFINE_integer(
    'id_begin',
    0,
    ''
)

_ID_END = flags.DEFINE_integer(
    'id_end',
    2000,
    ''
)

_SPAN_LENGTH = flags.DEFINE_integer(
    'span_length',
    1,
    ''
)

_STORED_ID_BEGIN = flags.DEFINE_integer(
    'stored_id_begin',
    0,
    ''
)

_SAVE_DIR = flags.DEFINE_string(
    'save_dir',
    '<YOUR_DIR>',
    ''
)

_SAVE_STEPS = flags.DEFINE_integer(
    'save_steps',
    10,
    ''
)

_SEED = flags.DEFINE_integer(
    'seed', 2024, '')


def dataset_generate_neighbors(dataset, mask_model, mask_tokenizer, span_length, pct, neighbour_num, id_begin, stored_id_begin, save_path, save_steps, ceil_pct=True):
    id = id_begin
    if id == stored_id_begin:
        neighbours = []
    else:
        neighbours = load_json_file(save_path)
    
    dataset_ = [x['input'] for x in dataset]
    
    for i in tqdm(range(len(dataset_))):
        input_text = dataset_[i]
        
        input_neigbour = [''.join(perturb_texts(input_text, mask_model, mask_tokenizer, span_length, pct, ceil_pct=True)) for _ in range(neighbour_num)]
        
        temp_dict = {
            'id': id,
            'ori_id': dataset[id-id_begin]['id'],
            'ori_input': dataset[id-id_begin]['input'],
            'input_neighbour_batch': input_neigbour,
        }
        id += 1
        neighbours.append(temp_dict)
        
        if id % save_steps == 0:
            save_json_file(save_path, neighbours)
        
    return neighbours

def main(_):
    seed_everything(_SEED.value)
    
    print("------------------ 1.load model ---------------------")
    mask_model = AutoModelForSeq2SeqLM.from_pretrained(_MASK_MODEL_DIR.value, device_map="cuda:0")
    mask_tokenizer =  AutoTokenizer.from_pretrained(_MASK_MODEL_DIR.value, model_max_length=512)
    print("complete")
    
    
    print("------------------ 2.load member & nonmember data  ---------------------")
    member_list = load_json_file(_MEMBER_PATH.value)
    nonmember_list = load_json_file(_NONMEMBER_PATH.value)
    data_list = member_list + nonmember_list
    print("complete")
    
    
    print("------------------ 3.data augmentation ---------------------")
    os.makedirs(_SAVE_DIR.value, exist_ok=True)
    save_path = _SAVE_DIR.value + "/" + "perturb_" + str(_STORED_ID_BEGIN.value) + "-" + str(_ID_END.value) + "_neighbour" + str(_NEIGHBOUR_NUMS.value) + "_pc" + str(int(_PERTURB_PC.value*100)) + ".json"
    print(f">>> save_path: {save_path}")
    
    neighbour_list = dataset_generate_neighbors(
        data_list[_ID_BEGIN.value: _ID_END.value], 
        mask_model, 
        mask_tokenizer, 
        _SPAN_LENGTH.value, 
        _PERTURB_PC.value, 
        _NEIGHBOUR_NUMS.value, 
        _ID_BEGIN.value,
        _STORED_ID_BEGIN.value, 
        save_path, 
        _SAVE_STEPS.value, 
        ceil_pct=True
    )
    
    
    save_json_file(save_path, neighbour_list)
    
if __name__ == '__main__':
    app.run(main)