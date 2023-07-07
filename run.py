import pandas as pd
import pickle
import os
from simpletransformers.ner import NERModel
from utils.args import GecArgs

output_base_dir = os.path.join(os.path.abspath('.'), 'outputs')

args = GecArgs.args

from utils import BIO_labels
labels = BIO_labels
if args.model_type != 'interactive':
    labels = labels[:-1]

train_file, eval_file, test_file = args.train_dev_data

print("reading train file:", train_file)
with open(train_file, 'rb') as f:
    train_data = pickle.load(f)
print("reading eval file:", eval_file)
with open(eval_file, 'rb') as f:
    eval_data = pickle.load(f)
print("reading test file:", test_file)
with open(test_file, 'rb') as f:
    test_data = pickle.load(f)

if args.debug:
    train_data = train_data[:[i for i, d in enumerate(train_data) if d[0]<6][-1]]
    eval_data = eval_data[:[i for i, d in enumerate(eval_data) if d[0]<6][-1]]
    test_data = test_data[:[i for i, d in enumerate(test_data) if d[0]<6][-1]]
    wandb_project = False
else:
    wandb_project = "gec_ner"

columns=["sentence_id", "words", "labels", "cls_labels", "correction_index", "parsing_embedding"]

train_df = pd.DataFrame(train_data, columns=columns)
eval_df = pd.DataFrame(eval_data, columns=columns)
if test_file:
    test_df = pd.DataFrame(test_data, columns=columns)
if not args.parsing_embedding:
    train_df = train_df.drop(['parsing_embedding'], axis=1)
    eval_df = eval_df.drop(['parsing_embedding'], axis=1)
    test_df = test_df.drop(['parsing_embedding'], axis=1)

print(len(train_df), len(eval_df))
print(len(test_df))

if args.only_inference is not None:
    args.model_name = output_base_dir + args.exp_name
print(args.exp_name)

if args.only_inference is not None:
    if args.output_dir is None:
        output_dir = output_base_dir + args.exp_name
    else:
        output_dir = args.output_dir
        args.exp_name = output_dir
else:
    output_dir = output_base_dir + args.exp_name + '/eval'

model_args = {"overwrite_output_dir": True,
          "num_train_epochs": args.epochs,
          "train_batch_size": args.train_batch_size,
          "eval_batch_size": args.eval_batch_size,
          "output_dir": output_dir,
          "reprocess_input_data": True,
          "special_tokens_list": ["[NONE]", "[MOD]"],
          "wandb_kwargs": {
              "mode": 'offline',
              "name": args.exp_name,
          },
          "wandb_project": wandb_project,
          "evaluate_during_training": args.evaluate_during_training,
          "evaluate_each_epoch": args.evaluate_each_epoch,
          "learning_rate": args.lr,
          "multi_loss": args.multi_loss,
          "wo_token_labels": args.wo_token_labels,
          "cls_num_labels": 15, # label nums for [CLS] token classification
          "use_multiprocessing_for_evaluation": False,
          "use_multiprocessing": args.use_multiprocessing,
          "loss_weight": args.loss_weight,
          "max_correction_embeddings": args.max_correction_embeddings,
          "max_seq_length": args.max_seq_length,
          "n_gpu": args.n_gpu,
          "dataloader_num_workers": 20,
          "save_eval_checkpoints": False,
          "early_stopping_metric": "f1_score",
          "best_model_dir": output_dir,
          "parsing_embedding": args.parsing_embedding,
          "parsing_embedding_for_embedding": args.parsing_embedding_for_embedding,
          "logging_steps": 0,
          "manual_seed": 42
          }

model = NERModel(
    model_type=args.model,
    model_name=args.model_name,
    labels=labels,
    args=model_args,
)

# Create a NERModel
if args.only_inference is None:
    # Train the model
    model.train_model(train_df, eval_data=eval_df, test_data=test_df)

if args.only_inference is None or 'dev' in args.only_inference:
    result, model_outputs, predictions, out_label_list = model.eval_model(eval_df, wandb_log=False)
if args.only_inference is None or 'test' in args.only_inference:
    result, model_outputs, predictions, out_label_list = model.eval_model(test_df, wandb_log=False)
