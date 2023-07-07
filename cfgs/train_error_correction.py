config = {
    "exp_name" : '/error_correction/ner_1e5_bs32_bert_large',
    "eval_res_dir": '',
    "only_inference" : None,
    "train_dev_data": ['data/ner/train.pkl',
                       'data/ner/dev.pkl',
                       'data/ner/test.pkl',],
    "max_correction_embeddings": 0, # set 0 if no correction embeddings, else 3
    "lr": 1e-5,
    "debug": False,
    "max_seq_length": 256,
    "model_type": 'non_interactive'
}
