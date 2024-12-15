from absl import app
from absl import flags
from transformers import TrainingArguments
from trl import SFTTrainer

from utils import load_model_and_tokenizer, load_json_file


_TRAIN_SET_PATH = flags.DEFINE_string(
    'train_set_path',
    '<YOUR_PATH>',
    'path of training file'
)

_MODEL_PATH = flags.DEFINE_string(
    'model_path',
    '<YOUR_PATH>',
    'sft model path'
)

_USE_FLASH_ATTN = flags.DEFINE_bool(
    'use_flash_attention',
    False,
    ''
)

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    '<YOUR_DIR>',
    'sft model path'
)

_DATASET_NAME =  flags.DEFINE_string(
    'dataset_name',
    '<NAME>',
    ''
)

_SFT_NAME = flags.DEFINE_string(
    'sft_name',
    '<NAME>',
    ''
)

_EPOCHS = flags.DEFINE_integer(
    'epochs',
    1,
    ''
)

_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate',
    5e-5,
    ''
)


def prompt_format(prompt):
    return prompt

def main(_):
    training_dataset = load_json_file(_TRAIN_SET_PATH.value)
    model, tokenizer = load_model_and_tokenizer(_MODEL_PATH.value, _USE_FLASH_ATTN.value)
    
    save_dir = _OUTPUT_DIR.value + _DATASET_NAME.value
    
    model_output_dir = save_dir + "/" + _SFT_NAME.value
    
    args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=_EPOCHS.value,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=_LEARNING_RATE.value,
        bf16=True,
        tf32=True,
        lr_scheduler_type="constant",
        disable_tqdm=True,
    )
    max_seq_length = 2048
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=training_dataset,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=prompt_format,
        args=args,
    )
    
    trainer.train()
    
if __name__ == "__main__":
    app.run(main)