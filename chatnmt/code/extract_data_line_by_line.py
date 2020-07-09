import json
import re


def chars_match(strg, search=re.compile(r'[^ <>c‖]').search):
    return not bool(search(strg))

# Opening JSON file


def extract_data(json_file_path, extracted_file_path, add_remove_boundry=False, delete_customer=False, delete_agent=False, add_context_symbol=True, is_testing=False):
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
    # prev_speaker = None
    en_tmp = ""
    de_tmp = ""
    for chat in chats.values():
        for turn in chat:
            if turn['speaker'] == "agent":
                if delete_agent == False:
                    en_tmp = turn['source'] 
                    de_tmp = turn['target'] 
                else:
                    en_tmp = ''
                    de_tmp = ''
            else:
                if delete_customer == False:
                    if add_context_symbol:
                        en_tmp = '<c> '+turn['target'] 
                        de_tmp = '<c> '+turn['source'] 
                    else:
                        en_tmp = turn['target'] 
                        de_tmp = turn['source'] 
                else:
                    en_tmp = ''
                    de_tmp = ''
            if en_tmp:
                if chars_match(en_tmp):
                    f_en.write("")
                    f_en.write('\n')
                else:
                    # print(en_tmp)
                    f_en.write(en_tmp)
                    f_en.write('\n')
            if de_tmp:
                if de_tmp == '<c> ':
                    f_en.write("")
                    f_en.write('\n')
                else:
                    f_de.write(de_tmp)
                    f_de.write('\n')
    

        if add_remove_boundry:
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
    extract_data(json_file_path, extracted_file_path,
                 add_remove_boundry=False, add_context_symbol=False)

    # generate dev file aligned line by line
    json_file_path = "../origdata/dev.json"
    extracted_file_path = "../data/dev"
    extract_data(json_file_path, extracted_file_path,
                 add_remove_boundry=False, delete_customer=True)

    # generate test file aligned line by line
    json_file_path = "../origdata/test.json"
    extracted_file_path = "../data/test"
    extract_data(json_file_path, extracted_file_path,
                 add_remove_boundry=False, delete_customer=True, is_testing=True)


def preprocess_boundaries():
    json_file_path = "../origdata/train.json"
    extracted_file_path = "../data_boundaries/train"
    # delete_customer = False
    extract_data(json_file_path, extracted_file_path,
                 add_remove_boundry=True, add_context_symbol=False)

    # generate dev file aligned line by line
    json_file_path = "../origdata/dev.json"
    extracted_file_path = "../data_boundaries/dev"
    extract_data(json_file_path, extracted_file_path,
                 add_remove_boundry=True, delete_customer=False)

    # generate test file aligned line by line
    json_file_path = "../origdata/test.json"
    extracted_file_path = "../data_boundaries/test"
    extract_data(json_file_path, extracted_file_path,
                 add_remove_boundry=True, delete_customer=False, is_testing=True)


def preprocess_iwlstfiles():
    # generate training file and testing file aligned line by line
    # testing file is split from training file, the ration is about 9:1
    json_file_path = "../origdata/train.json"
    extracted_file_path = "../dataiwlst/train"
    # delete_customer = False
    extract_data(json_file_path, extracted_file_path,
                 add_remove_boundry=False, add_context_symbol=False)

    # generate dev file aligned line by line
    json_file_path = "../origdata/dev.json"
    extracted_file_path = "../dataiwlst/dev"
    extract_data(json_file_path, extracted_file_path, add_remove_boundry=False,
                 delete_customer=False, delete_agent=True, add_context_symbol=False)

    # generate test file aligned line by line
    json_file_path = "../origdata/test.json"
    extracted_file_path = "../dataiwlst/test"
    extract_data(json_file_path, extracted_file_path, add_remove_boundry=False,
                 delete_customer=False, delete_agent=True, is_testing=False)


def main():
    preprocess_noboundaries()
    preprocess_boundaries()
    preprocess_iwlstfiles()


main()
