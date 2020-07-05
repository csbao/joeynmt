# Experimental results
https://docs.google.com/spreadsheets/d/1iboH-nYyNMLNKnjPehIm3ZhZ7VUrpQBcX7Ngid32KYw/edit?usp=sharing

# Note on Config & Model Directory for Submission
- We are using the official train/dev/split.
- We use config in `configs/wmt20/<MODEL_NAME>.yaml` and set `model_dir: "models/wmt20/<MODEL_NAME>/"` in that config. **Please make config filename & model directory name consistent**.
- We **load pre-trained model's vocab no matter it's fine-tuning or training from scratch**, to unify things in terms of # trainable parameters.
```
src_vocab: "models/wmt_ende_transformer/src_vocab.txt"
trg_vocab: "models/wmt_ende_transformer/trg_vocab.txt"
```
- Following the above, we always set `tied_embeddings: True` .
- Set `epochs: 50` because this might be enough.
