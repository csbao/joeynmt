import yaml

with open('/Users/calvinbao/Documents/src/spring2020/cmsc828b/joey/configs/viznet/1.yaml') as file:
    doc = yaml.load(file, Loader=yaml.FullLoader)
    features = {
        'num_heads_enc': [2,4, 8],
        'num_heads_dec': [2, 4, 8],
        'dropout': [0.1,0.2], # will be true for both enc and dec
        'hidden_size_enc': [64,128],
        'hidden_size_dec': [64,128],
        'batch_size': [64,128,256]
    }
    for num_head_enc in features['num_heads_enc']:
        for num_head_dec in features['num_heads_dec']:
            for dropout in features['dropout']:
                for hs_enc in features['hidden_size_enc']:
                    for hs_dec in features['hidden_size_dec']:
                        for bs in features['batch_size']:
                            doc["model"]["encoder"]["num_heads"] = num_head_enc
                            doc["model"]["decoder"]["num_heads"] = num_head_dec

                            doc["model"]["encoder"]["dropout"] = dropout
                            doc["model"]["encoder"]["hidden_size"] = hs_enc
                            doc["model"]["encoder"]["embeddings"]["embedding_dim"] = hs_enc
                            doc["model"]["decoder"]["hidden_size"] = hs_dec
                            doc["model"]["decoder"]["embeddings"]["embedding_dim"] = hs_dec
                            doc["training"]["batch_size"] = bs
                            doc["training"]["model_dir"] = f"models/viznet/{num_head_enc}_{num_head_dec}_{dropout}_{hs_enc}_{hs_dec}_{bs}"
                            with open(f"/Users/calvinbao/Documents/src/spring2020/cmsc828b/joey/configs/viznet/{num_head_enc}_{num_head_dec}_{dropout}_{hs_enc}_{hs_dec}_{bs}.yml", 'w') as outfile:
                                yaml.dump(doc, outfile,default_flow_style=False, sort_keys=True)




        # self.num_heads_enc = config["model"]["encoder"]["num_heads"]
        # self.num_heads_dec = config["model"]["decoder"]["num_heads"]
        # self.dropout = config["model"]["encoder"]["dropout"]
        # self.hidden_size_enc = config["model"]["encoder"]["hidden_size"]
        # self.hidden_size_dec = config["model"]["decoder"]["hidden_size"]
