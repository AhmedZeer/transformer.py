def generate_config(tokenizer):

    return {
        "vocab_size" : tokenizer.get_vocab_size(),
        "d_model" : 1024,
        "seq_len" : 512,
        "head_num": 8,
        "ffn" : 2048,
        "dropout_value": 0.1,
        "blocks_num" : 6,
        "batch_size" : 1,
        "n_epochs" : 3
    }
