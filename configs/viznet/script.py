import yaml
#   train: "chatnmt/multi_encoder/train.tags.bpe.10000"    # training data
#   dev: "chatnmt/multi_encoder/dev.tags.bpe.10000"        # development data for validation
#   test: "chatnmt/multi_encoder/test.tags.bpe.10000"      # test data for testing final model; optional
with open('/Users/calvinbao/Documents/src/spring2020/cmsc828b/joey_calvinbao94/configs/viznet/1.yaml') as file:
    doc = yaml.load(file, Loader=yaml.FullLoader)
    features = {
        'num_heads_enc': [ 8, 16],
        'num_heads_dec': [ 8, 16],
        # 'dropout': [0.1,0.2, 0.3], # will be true for both enc and dec
        'dropout': [0.1, 0.2, 0.4],
        # 'hidden_size_enc': [128,256], TODO: uncomment this after
        # 'hidden_size_dec': [128,256], TODO: uncomment this after
        'hidden_size_enc': [1024],
        'batch_size': [1024,2048],
        'enc_layers': [6, 8],
        'dec_layers': [6,8]
       
    }
    for num_head_enc in features['num_heads_enc']:
        for num_head_dec in features['num_heads_dec']:
            for enc_layer in features['enc_layers']:
                for dec_layer in features['dec_layers']:
                    for dropout in features['dropout']:
                        for hs_enc in features['hidden_size_enc']:
                            for bs in features['batch_size']:
                                doc["model"]["encoder"]["num_heads"] = num_head_enc
                                doc["model"]["decoder"]["num_heads"] = num_head_dec

                                doc["model"]["encoder"]["dropout"] = dropout
                                doc["model"]["encoder"]["hidden_size"] = hs_enc
                                doc["model"]["encoder"]["embeddings"]["embedding_dim"] = hs_enc
                                doc["model"]["encoder"]["multi_encoder"] = False

                                doc["model"]["decoder"]["hidden_size"] = hs_enc
                                doc["model"]["decoder"]["embeddings"]["embedding_dim"] = hs_enc
                                doc["training"]["batch_size"] = bs
                                doc["training"]["model_dir"] = f"models/viznet/attempt4/{num_head_enc}_{num_head_dec}_{dropout}_{hs_enc}_{hs_enc}_{bs}_{enc_layer}_{dec_layer}"
                                with open(f"/Users/calvinbao/Documents/src/spring2020/cmsc828b/joey_calvinbao94/configs/viznet/attempt4/{num_head_enc}_{num_head_dec}_{dropout}_{hs_enc}_{hs_enc}_{bs}_{enc_layer}_{dec_layer}.yml", 'w') as outfile:
                                    yaml.dump(doc, outfile,default_flow_style=False, sort_keys=True)