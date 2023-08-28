import pathlib
from transformers import BertTokenizer
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    LineByLineTextDataset
)
from transformers import BertConfig, BertForMaskedLM

VOCAB_SIZE = {4: 159000, 5: 591600, 6: 682400}
LEARNING_RATE = 0.001
BATCH_SIZE = 4
EPOCHS = 1000
PRECISION = 4

configuration = BertConfig(vocab_size = VOCAB_SIZE[PRECISION])
model = BertForMaskedLM(configuration).cuda()
tokenizer=BertTokenizer(vocab_file=f'./geo_vocab_{PRECISION}.txt',do_basic_tokenize = False)
train_dataset=LineByLineTextDataset(tokenizer=tokenizer,file_path=f'./traj_corpus_{PRECISION}.txt',block_size=128) 

training_args = TrainingArguments(
    output_dir='./checkpoints/', overwrite_output_dir=True, num_train_epochs=EPOCHS, learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE, save_total_limit=10)# save_steps=10000

print(f"finish configuration ...")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
print(f"finish data collation ...")


trainer = Trainer(
    model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset)

print(f"finish set up trainer ...")

if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    print("resume from existed checkpoints")
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()
# trainer.train(resume_from_checkpoint=checkpoint_dir)
trainer.save_model('./gpsbert-base')