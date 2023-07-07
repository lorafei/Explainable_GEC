import argparse
import importlib
import datetime

class GecArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to config file")
    parser.add_argument("--only_inference", type=list, default=None,
                        help="if is only inference and inference which data. Set None if is training",
                        choices=[None, 'train', 'test', 'dev'])
    parser.add_argument("--model", type=str, default='bert', help="if use a pretrained_model")
    parser.add_argument("--model_type", type=str, default='interactive', help="if with syntax",
                        choices=['interactive', 'noninteractive'])
    parser.add_argument("--model_name", type=str, default="bert-large-cased", help="if use a pretrained_model")
    parser.add_argument("--eval_res_dir",  type=str, default='', help="path to save the evaluation results")
    parser.add_argument("--only_eval", type=bool, default=False, help="")
    parser.add_argument("--split_dev", type=bool, default=False, help="if split dev to train set")
    parser.add_argument("--half_train", type=int, default=0, help="if reduce trainset to half")
    parser.add_argument("--half_eval", type=int, default=0, help="if reduce evalset to half")
    parser.add_argument("--exp_name", type=str, default='test', help="name of the exp")
    parser.add_argument("--output_dir", type=str, default=None, help="name of the exp")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="")
    parser.add_argument("--train_batch_size", type=int, default=8, help="")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="")
    parser.add_argument("--train_dev_data", type=list, default=[], help="")
    parser.add_argument("--multi_loss", type=bool, default=False, help="")
    parser.add_argument("--loss_weight", type=list, default=[0.5, 0.5], help="")
    parser.add_argument("--wo_token_labels", type=bool, default=False, help="")
    parser.add_argument("--debug", type=bool, default=False, help="")
    parser.add_argument("--labels_list", type=list, default=[], help="")
    parser.add_argument("--with_errant", type=bool, default=False, help="")
    parser.add_argument("--max_correction_embeddings", type=int, default=3,
                        help="if add extra correction position embedding for indicating the position of the correction")
    parser.add_argument("--interactive_mode", type=bool, default=False, help="")
    parser.add_argument("--rule_data", type=list, default=[], help="")
    parser.add_argument("--new_data", type=list, default=[], help="")
    parser.add_argument("--parallel", type=bool, default=False, help="parallel data, tgt+[SEP]+src")
    parser.add_argument("--max_seq_length", type=int, default=256, help="max_seq_length of BERT")
    parser.add_argument("--test_file", type=str, default='', help="pkl file for test")
    parser.add_argument("--n_gpu", type=int, default=1, help="gpu number for distributed training")
    parser.add_argument("--ensemble_reference", type=int, default=None, help="Ensembled reference")
    parser.add_argument("--use_multiprocessing", type=bool, default=False, help="")
    parser.add_argument("--optimizer", type=str, default='AdamW')
    parser.add_argument("--parsing_embedding", type=bool, default=False)
    parser.add_argument("--eval_all_checkpoint", type=bool, default=False)
    parser.add_argument("--evaluate_each_epoch", type=bool, default=True)
    parser.add_argument("--evaluate_during_training", type=bool, default=True)
    parser.add_argument("--stbert_ensemble", type=bool, default=False)
    parser.add_argument("--is_qk", type=bool, default=False)
    parser.add_argument("--is_dense", type=bool, default=False)
    parser.add_argument("--mn", type=int, default=2)
    parser.add_argument("--two_linears", type=int, default=0)
    parser.add_argument("--linear_hidden_size", type=int, default=1024)
    parser.add_argument("--parsing_heads", type=bool, default=False)
    parser.add_argument("--parsing_heads_reshape", type=str, default=None,)
    parser.add_argument("--parsing_embedding_matrix", type=bool, default=False)
    parser.add_argument("--parsing_embedding_for_embedding", type=bool, default=False)
    parser.add_argument("--parsing_embedding_for_model_matrix_embedding", type=bool, default=False)
    parser.add_argument("--only_first_parsing_order", type=bool, default=False)


    args = parser.parse_args()
    config_path = args.config.replace('/', '.')[:-3]

    config = importlib.import_module(config_path).config
    # time_str = datetime.now().strftime('%Y%m%d-%H%M%S')

    # update based on the config
    for k, v in config.items():
        if k not in ['custom']:
            setattr(args, k, v)
