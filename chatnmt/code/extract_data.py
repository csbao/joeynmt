import json
import re


def chars_match(strg, search=re.compile(r'[^ <>c‖]').search):
    return not bool(search(strg))

# Opening JSON file


def extract_data(json_file_path, extracted_file_path, add_remove_boundry=False, delete_customer=False,is_training=False,is_testing=False):
    # assert add_remove_boundry !=delete_customer
    # PATH = "../origdata/train.json"
    f = open(json_file_path,)
    en_file_path = extracted_file_path + '.en'
    de_file_path = extracted_file_path + '.de'
    f_en = open(en_file_path, 'w')
    f_de = open(de_file_path, 'w')
    turn_separator = ' ‖ ' 

    # returns JSON object as
    # a dictionary
    # opened = False
    chats = json.load(f)
    # i = 0
    for chat in chats.values():
        # i = i + 1
        # if split_training:
        #     if i==495 :
        #         f_en.close()
        #         f_de.close()
        #         f_en = open(en_file_path[:-8] +'test' + '.en','w+')
        #         f_de = open(en_file_path[:-8] +'test' + '.de','w+')
        wrote = False
        prev_speaker = None
        en_tmp = ""
        de_tmp = ""
        for turn in chat:
            if turn['speaker'] != prev_speaker:
                wrote = True
                if en_tmp:
                    if chars_match(en_tmp):
                        f_en.write("")
                        f_en.write('\n')
                    else:
                        if en_tmp.endswith(turn_separator):
                            en_tmp = en_tmp[:-3]
                        # print(en_tmp)
                        f_en.write(en_tmp)
                        f_en.write('\n')
                if de_tmp:
                    if de_tmp=='<c> ':
                        f_en.write("")
                        f_en.write('\n')
                    else:
                        if de_tmp.endswith(turn_separator):
                            de_tmp = de_tmp[:-3]
                        f_de.write(de_tmp)
                        f_de.write('\n')
                if turn['speaker'] == "agent":
                    en_tmp = turn['source'] + turn_separator
                    de_tmp = turn['target'] + turn_separator
                else:
                    if delete_customer == False:
                        if not is_training:
                            en_tmp = '<c> '+turn['target'] + turn_separator
                            de_tmp = '<c> '+turn['source'] + turn_separator
                        else:
                            en_tmp = turn['target'] + turn_separator
                            de_tmp = turn['source'] + turn_separator
                    else:
                        en_tmp = ''
                        de_tmp = ''
            else:
                if turn['speaker'] == "agent":
                    en_tmp += turn['source'] + turn_separator
                    de_tmp += turn['target'] + turn_separator
                else:
                    if delete_customer == False:
                        en_tmp += turn['target'] + turn_separator
                        de_tmp += turn['source'] + turn_separator
                    else:
                        en_tmp = ''
                        de_tmp = ''
            prev_speaker = turn['speaker']
        # print end of convo, for boundaries

        if wrote and add_remove_boundry:
            f_en.write('REMOVEMEIMABOUNDARY\n')
            f_de.write('REMOVEMEIMABOUNDARY\n')
            # pass
    # print(i)
    if is_testing:
        f_de.seek(0)
        f_de.truncate()
    f.close()
    f_en.close()
    f_de.close()


def preprocess_noboundaries():
    # generate training file and testing file aligned line by line
    # testing file is split from training file, the ration is about 9:1
    json_file_path = "../origdata/train.json"
    extracted_file_path = "../data/train"
    # delete_customer = False
    extract_data(json_file_path, extracted_file_path,add_remove_boundry=False,is_training=True)

    # generate dev file aligned line by line
    json_file_path = "../origdata/dev.json"
    extracted_file_path = "../data/dev"
    extract_data(json_file_path, extracted_file_path,add_remove_boundry=False, delete_customer=True)

    # generate test file aligned line by line
    json_file_path = "../origdata/test.json"
    extracted_file_path = "../data/test"
    extract_data(json_file_path, extracted_file_path,add_remove_boundry=False, delete_customer=True,is_testing=True)

def preprocess_boundaries():
    json_file_path = "../origdata/train.json"
    extracted_file_path = "../data_boundaries/train"
    # delete_customer = False
    extract_data(json_file_path, extracted_file_path,add_remove_boundry=True,is_training=True)

    # generate dev file aligned line by line
    json_file_path = "../origdata/dev.json"
    extracted_file_path = "../data_boundaries/dev"
    extract_data(json_file_path, extracted_file_path,add_remove_boundry=True, delete_customer=False)

    # generate test file aligned line by line
    json_file_path = "../origdata/test.json"
    extracted_file_path = "../data_boundaries/test"
    extract_data(json_file_path, extracted_file_path,add_remove_boundry=True, delete_customer=False,is_testing=True)

def main():
    preprocess_noboundaries()
    preprocess_boundaries()



main()
