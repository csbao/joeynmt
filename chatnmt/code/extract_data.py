import json

# Opening JSON file
def extract_data(json_file_path,extracted_file_path, split_training=True, add_remove_boundry=True):
    # PATH = "../origdata/train.json"
    f = open(json_file_path,)
    en_file_path = extracted_file_path + '.en'
    de_file_path = extracted_file_path + '.de'
    f_en = open(en_file_path,'w')
    f_de = open(de_file_path,'w')

    # returns JSON object as
    # a dictionary
    # opened = False
    chats = json.load(f)
    i = 0
    for chat in chats.values():
        i = i + 1
        if split_training:
            if i==495 :
                f_en.close()
                f_de.close()
                f_en = open(en_file_path[:-8] +'test' + '.en','w+')
                f_de = open(en_file_path[:-8] +'test' + '.de','w+')
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

        if wrote and add_remove_boundry:
            f_en.write('REMOVEMEIMABOUNDARY\n')
            f_de.write('REMOVEMEIMABOUNDARY\n')
            # pass
    # print(i)
    f.close()
    f_en.close()
    f_de.close()


def main():
    #generate training file and testing file aligned line by line
    # testing file is split from training file, the ration is about 9:1
    json_file_path = "../origdata/train.json"
    extracted_file_path = "../data/train"
    extract_data(json_file_path, extracted_file_path)

    #generate dev file aligned line by line
    json_file_path = "../origdata/dev.json"
    extracted_file_path = "../data/dev"
    extract_data(json_file_path, extracted_file_path,split_training = False)


main()
