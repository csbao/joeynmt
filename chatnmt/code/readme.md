Firstly, `cd` to the folder chatnmt/code.   

The script for extracting data from the json file in the folder origddata to line-alignment data file 
in the folder data is `python extract_data.py` after that data would be put in the data/{train,dev,test}.{en,de}. The original training data is split into training data and test data based on conversions and the ratio is 9:1. There are 550 conversions in the original training data and the first 495 conversions 
are selected as training data and the remaining 55 conversions are selected as test data.

The script for preprocessing data is 
```sh preprocess.sh```
 basically, it would normalize the data, remove non-printable chars, tokenize the data, and the apply bpe model to the data.
