config = {
    "exp_name" : '/error_correction_ce_syntax/ner_1e5_bs32_bert_large/checkpoint-n', # best checkpoint path
    "eval_res_dir" : 'eval/dev.pkl',
    "only_inference" : ['dev'], # choose from ['dev', 'test'] or both
    "train_dev_data": ['data/ner/train.pkl',
                       'data/ner/dev.pkl',
                       'data/ner/test.pkl',],
    "debug": False,
    "max_seq_length": 256,
    "parsing_embedding": True,
    "parsing_embedding_for_embedding": True,
    "model_type": 'non_interactive'
}
