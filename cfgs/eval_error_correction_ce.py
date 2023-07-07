config = {
    "exp_name" : '/error_correction_ce/ner_1e5_bs32_bert_large/checkpoint-n', # best checkpoint path
    "eval_res_dir" : 'eval/dev.pkl',
    "only_inference" : ['dev'], # choose from ['dev', 'test'] or both
    "train_dev_data": ['data/ner/train.pkl',
                       'data/ner/dev.pkl',
                       'data/ner/test.pkl',],
    "debug": False,
    "max_seq_length": 256,
    "model_type": 'non_interactive'
}
