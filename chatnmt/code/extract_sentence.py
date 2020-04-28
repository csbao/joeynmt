# import nltk
# nltk.download('punkt')
import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')
fp = open("/Users/song/Desktop/nmt/joeynmt/chatnmt/data/test.de")
fout = open("/Users/song/Desktop/nmt/joeynmt/chatnmt/prep/test_concat_prev.de",'w')
line = fp.readline()
sentences = tokenizer.tokenize(line)
while True:
    last_sentence = sentences[-1]
    for i in range(0,len(sentences)-1):
        fout.write(sentences[i]+'<CONCAT>'+sentences[i+1])
    line = fp.readline()
    if not line:
        break
    sentences = tokenizer.tokenize(line)
    fout.write(last_sentence+'<CONCAT>'+sentences[0])
    fout.write("\n")
fp.close()
fout.close()
    
