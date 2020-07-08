# coding: utf-8
"""
Data module
"""
import sys
import random
import os, io
import os.path
from typing import Optional

from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field

from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN, CONTEXT_TOKEN, CONTEXT_EOS_TOKEN
from joeynmt.vocabulary import build_vocab, Vocabulary



def load_data(data_cfg: dict, multi_encoder: bool = False) -> (Dataset, Dataset, Optional[Dataset],
                                                          Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    If you set ``random_train_subset``, a random selection of this size is used
    from the training set instead of the full training set.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - src_vocab: source vocabulary extracted from training data
        - trg_vocab: target vocabulary extracted from training data
    """
    # load data from files
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg.get("test", None)
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    tok_fun = lambda s: list(s) if level == "char" else s.split()

    src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)

    trg_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           unk_token=UNK_TOKEN,
                           batch_first=True, lower=lowercase,
                           include_lengths=True)

    prev_src_field = data.Field(init_token=CONTEXT_TOKEN, eos_token=CONTEXT_EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)

    prev_trg_field = data.Field(init_token=CONTEXT_TOKEN, eos_token=CONTEXT_EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           unk_token=UNK_TOKEN,
                           batch_first=True, lower=lowercase,
                           include_lengths=True)
    train_data = None


    if multi_encoder:
        train_data = ContextTranslationDataset(path=train_path,
                                               exts=("." + src_lang, "." + trg_lang),
                                               fields=(prev_src_field, prev_trg_field, src_field, trg_field),
                                               filter_pred=
                                               lambda x: len(vars(x)['src'])
                                                <= max_sent_length
                                                    and len(vars(x)['trg'])
                                                <= max_sent_length)
    else:
        train_data = TranslationDataset(path=train_path,
                                        exts=("." + src_lang, "." + trg_lang),
                                        fields=(src_field, trg_field),
                                        filter_pred=
                                        lambda x: len(vars(x)['src'])
                                        <= max_sent_length
                                        and len(vars(x)['trg'])
                                        <= max_sent_length)

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    src_vocab_file = data_cfg.get("src_vocab", None)
    trg_vocab_file = data_cfg.get("trg_vocab", None)

    src_vocab = build_vocab(field="src", min_freq=src_min_freq,
                            max_size=src_max_size,
                            dataset=train_data, vocab_file=src_vocab_file)
    trg_vocab = build_vocab(field="trg", min_freq=trg_min_freq,
                            max_size=trg_max_size,
                            dataset=train_data, vocab_file=trg_vocab_file)
    prev_src_vocab = None
    prev_trg_vocab = None
    if multi_encoder:
        prev_src_vocab = build_vocab(field="src_prev", min_freq=src_min_freq,
                                        max_size=src_max_size,
                                        dataset=train_data, vocab_file=src_vocab_file)

        prev_trg_vocab = build_vocab(field="trg_prev", min_freq=trg_min_freq,
                                max_size=trg_max_size,
                                dataset=train_data, vocab_file=trg_vocab_file)
    random_train_subset = data_cfg.get("random_train_subset", -1)
    if random_train_subset > -1:
        # select this many training examples randomly and discard the rest
        keep_ratio = random_train_subset / len(train_data)
        keep, _ = train_data.split(
            split_ratio=[keep_ratio, 1 - keep_ratio],
            random_state=random.getstate())
        train_data = keep
    dev_data = None
    if multi_encoder:
        dev_data = ContextTranslationDataset(path=dev_path,
                                               exts=("." + src_lang, "." + trg_lang),
                                               fields=(prev_src_field, prev_trg_field, src_field, trg_field),
                                               filter_pred=
                                               lambda x: len(vars(x)['src'])
                                                <= max_sent_length
                                                    and len(vars(x)['trg'])
                                                <= max_sent_length)
    else:
        dev_data = TranslationDataset(path=dev_path,
                                      exts=("." + src_lang, "." + trg_lang),
                                      fields=(src_field, trg_field))
    test_data = None
    if test_path is not None:
        # check if target exists
        if os.path.isfile(test_path + "." + trg_lang):
            if multi_encoder:
                test_data = ContextTranslationDataset(path=test_path,
                                                       exts=("." + src_lang, "." + trg_lang),
                                                       fields=(prev_src_field, prev_trg_field, src_field, trg_field),
                                                       filter_pred=
                                                       lambda x: len(vars(x)['src'])
                                                        <= max_sent_length
                                                            and len(vars(x)['trg'])
                                                        <= max_sent_length)
            else:
                test_data = TranslationDataset(
                    path=test_path, exts=("." + src_lang, "." + trg_lang),
                    fields=(src_field, trg_field))
        else:
            # no target is given -> create dataset from src only
            test_data = MonoDataset(path=test_path, ext="." + src_lang,
                                    field=src_field)
    src_field.vocab = src_vocab
    trg_field.vocab = trg_vocab

    prev_src_field.vocab = prev_src_vocab
    prev_trg_field.vocab = prev_trg_vocab
    return train_data, dev_data, test_data, src_vocab, trg_vocab


# pylint: disable=global-at-module-level
global max_src_in_batch, max_tgt_in_batch


# pylint: disable=unused-argument,global-variable-undefined
def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0

    max_src_in_batch = max(max_src_in_batch, len(new.src))
    src_elements = count * max_src_in_batch
    if hasattr(new, 'trg'):  # for monolingual data sets ("translate" mode)
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
        tgt_elements = count * max_tgt_in_batch
    else:
        tgt_elements = 0
    return max(src_elements, tgt_elements)


def make_data_iter(dataset: Dataset,
                   batch_size: int,
                   batch_type: str = "sentence",
                   train: bool = False,
                   shuffle: bool = False) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        # optionally shuffle and sort during training
        data_iter = data.BucketIterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=shuffle)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.BucketIterator(
            repeat=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=False, sort=False)

    return data_iter


class MonoDataset(Dataset):
    """Defines a dataset for machine translation without targets."""

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, path: str, ext: str, field: Field, **kwargs) -> None:
        """
        Create a monolingual dataset (=only sources) given path and field.

        :param path: Prefix of path to the data file
        :param ext: Containing the extension to path for this language.
        :param field: Containing the fields that will be used for data.
        :param kwargs: Passed to the constructor of data.Dataset.
        """

        fields = [('src', field)]

        if hasattr(path, "readline"):  # special usage: stdin
            src_file = path
        else:
            src_path = os.path.expanduser(path + ext)
            src_file = open(src_path)

        examples = []
        for src_line in src_file:
            src_line = src_line.strip()
            if src_line != '':
                examples.append(data.Example.fromlist(
                    [src_line], fields))

        src_file.close()

        super(MonoDataset, self).__init__(examples, fields, **kwargs)

class ContextTranslationDataset(Dataset):
    """ Defines a dataset for nmt context translation."""
    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, path: str, exts, fields, **kwargs) -> None:
        """
        Create a dataset given path and field.

        :param path: Prefix of path to the data file
        :param ext: Containing the extension to path for this language.
        :param fields: Containing the fields that will be used for data.
        :param kwargs: Passed to the constructor of data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src_prev', fields[0]), ('trg_prev', fields[1]), ('src', fields[2]), ('trg', fields[3])]
        # fields = [('src', field)]
        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)
        examples = []

        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
                io.open(trg_path, mode='r', encoding='utf-8') as trg_file:
            prev_src_line, prev_trg_line = CONTEXT_TOKEN + CONTEXT_EOS_TOKEN, CONTEXT_TOKEN + CONTEXT_EOS_TOKEN
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if src_line != '' and trg_line != '':
                    if src_line == 'REMOVEMEIMABOUNDARY':
                        assert trg_line == 'REMOVEMEIMABOUNDARY', "TRG IS NOT A LINE"
                        prev_src_line, prev_trg_line = CONTEXT_TOKEN + CONTEXT_EOS_TOKEN, CONTEXT_TOKEN + CONTEXT_EOS_TOKEN
                        continue
                    if len(src_line) > 3 and src_line[:4] == (CONTEXT_TOKEN + " "):
                        prev_src_line, prev_trg_line = src_line[4:], trg_line[4:]
                        continue
                        # prev_src_line, prev_trg_line = CONTEXT_TOKEN + CONTEXT_EOS_TOKEN, CONTEXT_TOKEN + CONTEXT_EOS_TOKEN
                    examples.append(data.Example.fromlist(
                        [prev_src_line, prev_trg_line, src_line, trg_line], fields))
                    prev_src_line = src_line # Prepend special token, since src and context encoders share parameters
                    prev_trg_line = trg_line
        super(ContextTranslationDataset, self).__init__(examples, fields, **kwargs)
