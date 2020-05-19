import json

# Opening JSON file
PATH = "../origdata/train.json"
f = open(PATH,)
en_file_path = "../data/train.en"
de_file_path = "../data/train.de"

f_en = open(en_file_path,'w')
f_de = open(de_file_path,'w')



# returns JSON object as
# a dictionary
chats = json.load(f)

# Iterating through the json
# list
for chat in chats.values():
    wrote = False
    prev_speaker = chat[0]['speaker']
    en_tmp = ""
    de_tmp = ""
    for turn in chat:
        if turn['speaker']!=prev_speaker:
            wrote = True
            f_en.write(en_tmp)
            f_en.write('\n')
            f_de.write(de_tmp)
            f_de.write('\n')
            if turn['speaker']=="agent":
                en_tmp = turn['source']
                de_tmp = turn['target']
            else:
                en_tmp = turn['target']
                de_tmp = turn['source']
        else:
            if turn['speaker']=="agent":
                en_tmp+=turn['source']
                de_tmp+=turn['target']
            else:
                en_tmp+= turn['target']
                de_tmp+= turn['source']
        prev_speaker = turn['speaker']
    # print end of convo, for boundaries

    if wrote:
        f_en.write('\n')
        f_de.write('\n')
f.close()
f_en.close()
f_de.close()





# Closing file
f.close()
