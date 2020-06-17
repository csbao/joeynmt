#!/usr/bin/env bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh
# Adapted from https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt14.sh

################################################################
# This file will preprocess the files orig/train, orig/test, orig/valid and the output would be placed
# under prep. You may change the corresponding folders and files.
################################################################



# git clone https://github.com/moses-smt/mosesdecoder.git

MOSES=`pwd`/../../mosesdecoder

SCRIPTS=${MOSES}/scripts
TOKENIZER=${SCRIPTS}/tokenizer/tokenizer.perl
LC=${SCRIPTS}/tokenizer/lowercase.perl
CLEAN=${SCRIPTS}/training/clean-corpus-n.perl

merge_ops=iwslt14-deen-bpe
src=de
tgt=en
lang=de-en
prep="../multi_encoder"
# tmp=${prep}/tmp
orig="../data"
# prep = "../prep"
# train=train_concat_prev
# test=test_concat_prev
# dev=dev_concat_prev
train=train
test=test
dev=dev

# codes_file="../bpe/iwslt14-deen-bpe.32000"
codes_file="${prep}/bpe.${merge_ops}"

echo "pre-processing train data..."
for l in ${src} ${tgt}; do
    for p in ${train} ${dev} ${test}; do
        f=${p}.$l
        tok=${p}.tok.$l
        cat ${orig}/${f} | \
        perl ${TOKENIZER} -threads 8 -l $l > ${prep}/${tok}
        echo ""
    done
done
for p in ${train} ${dev} ${test}; do
    perl ${CLEAN} -ratio 9 ${prep}/${p}.tok ${src} ${tgt} ${prep}/${p}.clean 1 80
done
for l in ${src} ${tgt}; do
    for p in ${train} ${dev} ${test}; do
        perl ${LC} < ${prep}/${p}.clean.${l} > ${prep}/${p}.tags.${l}
    done
done

echo "learning * joint * BPE..."
codes_file="${prep}/bpe.${merge_ops}"
cat "${prep}/${train}.tags.${src}" "${prep}/${train}.tags.${tgt}" > ${prep}/${train}.tmp
python3 -m subword_nmt.learn_bpe -s "${merge_ops}" -i "${prep}/${train}.tmp" -o "${codes_file}"
# rm "${prep}/train.tmp"

echo "applying BPE..."
for l in ${src} ${tgt}; do
    for p in ${train} ${dev} ${test}; do
        python3 -m subword_nmt.apply_bpe -c "${codes_file}" -i "${prep}/${p}.tags.${l}" -o "${prep}/${p}.tags.bpe.${merge_ops}.${l}"
    done
done

# for l in ${src} ${tgt}; do
#     for p in train valid test; do
#         mv ${tmp}/${p}.${l} ${prep}/
#     done
# done

# mv "${codes_file}" "${prep}/"
# rm -rf ${MOSES}
# rm -rf ${tmp}