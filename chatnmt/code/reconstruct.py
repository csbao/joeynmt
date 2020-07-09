import json


def reconstruct(json_file_path, hyps_file_path):
    f_hyps = open(hyps_file_path, 'r')
    f_json = open(json_file_path,'r+')
    chats = json.load(f_json)
    lines = f_hyps.readline().split(' ‖ ')
    turn_idx = 0

    for chat in chats.values():
        for turn in chat:
            if turn['speaker'] == 'agent':
                if turn_idx >= len(lines):
                    lines = f_hyps.readline().split(' ‖ ')
                    turn_idx = 0
                
                assert(turn['target'] == shape_line(lines[turn_idx]))
                turn_idx += 1
    f_json.seek(0)
    json.dump(chats, f_json,ensure_ascii=False)
    f_json.truncate()

def shape_line(line):
    if line.endswith('\n'):
        line =  line[:-1]
    return line


def main():
    json_file_path = '/Users/song/Desktop/nmt/joeynmt/chatnmt/origdata/devtest.json'
    # hyps_file_path = '../../models/wmt20/tfm_b4096_ende/2000.hyps'
    hyps_file_path = '/Users/song/Desktop/nmt/joeynmt/chatnmt/data/dev.de'
    reconstruct(json_file_path, hyps_file_path)
main()
