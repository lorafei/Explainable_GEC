# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """

from __future__ import absolute_import, division, print_function
import enum
import linecache
import logging
import os
from io import open
from multiprocessing import Pool, cpu_count

import numpy as np

try:
    from collections import Iterable, Mapping
except ImportError:
    from collections.abc import Iterable, Mapping

import pandas as pd
import torch
from torch.functional import split
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from datasets import load_dataset
from datasets import Dataset as HFDataset


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(
            self,
            guid,
            words=None,
            source_words=None,
            target_words=None,
            labels=None,
            source_labels=None,
            target_labels=None,
            correction_index=None,
            source_correction_index=None,
            target_correction_index=None,
            parsing_ids=None,
            source_parsing_ids=None,
            target_parsing_ids=None,
            segment_ids=None,
            parsing_heads=None,
            x0=None,
            y0=None,
            x1=None,
            y1=None,
            tokenized_word_ids=None,
    ):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
            x0: (Optional) list. The list of x0 coordinates for each word.
            y0: (Optional) list. The list of y0 coordinates for each word.
            x1: (Optional) list. The list of x1 coordinates for each word.
            y1: (Optional) list. The list of y1 coordinates for each word.
            tokenized_word_ids: (Optional) list. Tokenized words converted to input_ids
        """
        self.guid = guid
        self.words = words
        self.source_words = source_words
        self.target_words = target_words

        self.labels = labels
        self.source_labels = source_labels
        self.target_labels = target_labels

        self.tokenized_word_ids = tokenized_word_ids

        self.correction_index = correction_index
        self.source_correction_index = source_correction_index
        self.target_correction_index = target_correction_index

        self.parsing_ids = parsing_ids
        self.source_parsing_ids = source_parsing_ids
        self.target_parsing_ids = target_parsing_ids

        self.segment_ids = segment_ids
        self.parsing_heads = parsing_heads

        if x0 is None:
            self.bboxes = None
        else:
            self.bboxes = [[a, b, c, d] for a, b, c, d in zip(x0, y0, x1, y1)]


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids,
                 bboxes=None, correction_index=None,
                 parsing_ids=None, stbert_segment_ids=None, parsing_heads=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.correction_index = correction_index
        self.parsing_ids = parsing_ids
        self.stbert_segment_ids = stbert_segment_ids
        self.parsing_heads = parsing_heads
        if bboxes:
            self.bboxes = bboxes


def read_examples_from_file(data_file, mode, bbox=False):
    file_path = data_file
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if bbox:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        examples.append(
                            InputExample(
                                guid="{}-{}".format(mode, guid_index),
                                words=words,
                                labels=labels,
                                x0=x0,
                                y0=y0,
                                x1=x1,
                                y1=y1,
                            )
                        )
                        guid_index += 1
                        words = []
                        labels = []
                        x0 = []
                        y0 = []
                        x1 = []
                        y1 = []
                else:
                    splits = line.split()
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[1])
                        x0.append(splits[2])
                        y0.append(splits[3])
                        x1.append(splits[4])
                        y1.append(splits[5])
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            else:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        examples.append(
                            InputExample(
                                guid="{}-{}".format(mode, guid_index),
                                words=words,
                                labels=labels,
                            )
                        )
                        guid_index += 1
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
        if words:
            if bbox:
                examples.append(
                    InputExample(
                        guid="%s-%d".format(mode, guid_index),
                        words=words,
                        labels=labels,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                    )
                )
            else:
                examples.append(
                    InputExample(
                        guid="%s-%d".format(mode, guid_index),
                        words=words,
                        labels=labels,
                    )
                )
    return examples


def get_examples_from_df(data, correction_index=None,
                         parsing_embedding=None):
    return_list = []

    for sentence_id, sentence_df in data.groupby(["sentence_id"]):
        words = sentence_df["words"].tolist()
        labels = sentence_df["labels"].tolist()
        input_seq_cls, input_correction_index, input_parsing_embedding = None, None, None
        if correction_index:
            input_correction_index = sentence_df["correction_index"].tolist()
        if parsing_embedding:
            input_parsing_embedding = sentence_df["parsing_embedding"].tolist()

        input_example = InputExample(
                guid=sentence_id,
                words=words,
                labels=labels,
                correction_index=input_correction_index,
                parsing_ids=input_parsing_embedding,
            )
        return_list.append(input_example)
    return return_list


def convert_example_to_feature(
        example,
        label_map,
        max_seq_length,
        tokenizer,
        cls_token_at_end,
        cls_token,
        cls_token_segment_id,
        sep_token,
        sep_token_extra,
        pad_on_left,
        pad_token,
        pad_token_segment_id,
        pad_token_label_id,
        sequence_a_segment_id,
        mask_padding_with_zero,
):
    def _convert_example_to_feature(example, label_map):
        tokens = []
        label_ids = []
        bboxes = []
        cor_ids = []
        parsing_ids = []

        parsing_heads = []
        pad_token_cor_id = 2

        pad_token_par_id = 4
        pad_token_parh_id = -100
        is_parsing = example.parsing_ids is not None
        is_correction = example.correction_index is not None
        is_parsing_head = example.parsing_heads is not None

        if isinstance(label_map, list):
            cls_label_map = label_map[1]
            label_map = label_map[0]
        else:
            cls_label_map = None
        for i, (word, label) in enumerate(zip(example.words, example.labels)):
            if example.tokenized_word_ids is None:
                word_tokens = tokenizer.tokenize(word)
            else:
                word_tokens = example.tokenized_word_ids[i]
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            if (
                    word_tokens
            ):  # avoid non printable character like '\u200e' which are tokenized as a void token ''
                tokens.extend(word_tokens)
            else:
                word_tokens = tokenizer.tokenize(tokenizer.unk_token)
                tokens.extend(word_tokens)
            label_ids.extend(
                [label_map[label]] + [pad_token_label_id] * (
                        len(word_tokens) - 1)
            )
            if is_correction:
                cor = example.correction_index[i]
                cor_ids.extend([cor] + [pad_token_cor_id] * (len(word_tokens) - 1))
            if is_parsing:
                par = example.parsing_ids[i]
                parsing_ids.extend([par] + [pad_token_par_id] * (len(word_tokens) - 1))
            if is_parsing_head:
                parh = example.parsing_heads[i]
                parsing_heads.extend([parh] + [pad_token_parh_id] * (len(word_tokens) - 1))
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            if bboxes:
                bboxes = bboxes[: (max_seq_length - special_tokens_count)]
            if is_correction:
                cor_ids = cor_ids[: (max_seq_length - special_tokens_count)]
            if is_parsing:
                parsing_ids = parsing_ids[: (max_seq_length - special_tokens_count)]
            if is_parsing_head:
                parsing_heads = parsing_heads[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if is_correction:
            cor_ids += [pad_token_cor_id]
        if is_parsing:
            parsing_ids += [pad_token_par_id]
        if is_parsing_head:
            parsing_heads += [pad_token_parh_id]

        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            if is_correction:
                cor_ids += [pad_token_cor_id]
            if is_parsing:
                parsing_ids += [pad_token_par_id]
            if is_parsing_head:
                parsing_heads += [pad_token_parh_id]

        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
            if is_correction:
                cor_ids += [pad_token_cor_id]
            if is_parsing:
                parsing_ids += [pad_token_par_id]
            if is_parsing_head:
                parsing_heads += [pad_token_parh_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids
            if is_correction:
                cor_ids = [pad_token_cor_id] + cor_ids
            if is_parsing:
                parsing_ids = [pad_token_par_id] + parsing_ids
            if is_parsing_head:
                parsing_heads += [pad_token_parh_id]

        if example.tokenized_word_ids is None:
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
        else:
            input_ids = tokens
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                                 [
                                     0 if mask_padding_with_zero else 1] * padding_length
                         ) + input_mask
            segment_ids = ([
                               pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            if is_correction:
                example.correction_index = ([
                                                pad_token_label_id] * padding_length) + example.correction_index
        else:
            segment_ids += [pad_token_segment_id] * padding_length
            if sep_token in tokens:
                segment_ids[0:tokens.index(sep_token) + 1] = [1] * (tokens.index(sep_token) + 1)
                segment_ids[tokens.index(sep_token) + 1:len(input_ids)] = [2] * (len(tokens) - tokens.index(sep_token) - 1)
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length

            label_ids += [pad_token_label_id] * padding_length
            if is_correction:
                cor_ids += [
                               pad_token_cor_id] * padding_length  # add [CLS] [SEP] in the front and end
            if is_parsing:
                parsing_ids += [pad_token_par_id] * padding_length
            if is_parsing_head:
                parsing_heads += [pad_token_parh_id] * padding_length

            stbert_segment_ids = None
        return input_ids, input_mask, segment_ids, label_ids, parsing_ids, cor_ids, is_correction, is_parsing, stbert_segment_ids, parsing_heads, is_parsing_head, cls_label_map


    input_ids, input_mask, segment_ids, label_ids, parsing_ids, cor_ids, \
    is_correction, is_parsing, stbert_segment_ids, parsing_heads, is_parsing_head, cls_label_map = _convert_example_to_feature(example, label_map)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    if is_correction:
        assert len(cor_ids) == max_seq_length
    if is_parsing:
        assert len(parsing_ids) == max_seq_length
    if is_parsing_head:
        assert len(parsing_heads) == max_seq_length

    correction_index, stbert_segment = None, None
    stbert_segment, parsing_heads_matrix, parsing= None, None, None
    if is_correction:
        correction_index = cor_ids
    if is_parsing_head:
        stbert_segment = stbert_segment_ids
        parsing_heads_matrix = parsing_heads
    if is_parsing:
        parsing = parsing_ids
        stbert_segment = stbert_segment_ids,
    return InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        correction_index=correction_index,
        stbert_segment_ids=stbert_segment,
        parsing_heads=parsing_heads_matrix,
        parsing_ids=parsing,
    )


def convert_examples_with_multiprocessing(examples):
    (
        example_groups,
        label_map,
        max_seq_length,
        tokenizer,
        cls_token_at_end,
        cls_token,
        cls_token_segment_id,
        sep_token,
        sep_token_extra,
        pad_on_left,
        pad_token,
        pad_token_segment_id,
        pad_token_label_id,
        sequence_a_segment_id,
        mask_padding_with_zero,
    ) = examples

    return [
        convert_example_to_feature(
            example,
            label_map,
            max_seq_length,
            tokenizer,
            cls_token_at_end,
            cls_token,
            cls_token_segment_id,
            sep_token,
            sep_token_extra,
            pad_on_left,
            pad_token,
            pad_token_segment_id,
            pad_token_label_id,
            sequence_a_segment_id,
            mask_padding_with_zero,
        )
        for example in example_groups
    ]


def convert_examples_to_features(
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-1,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        process_count=cpu_count() - 2,
        chunksize=500,
        silent=False,
        use_multiprocessing=True,
        mode="dev",
        use_multiprocessing_for_evaluation=False,
):
    """Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    if (mode == "train" and use_multiprocessing) or (
            mode == "dev" and use_multiprocessing_for_evaluation
    ):
        if chunksize == -1:
            chunksize = max(len(examples) // (process_count * 2), 500)
        new_examples = []

        for i in range(0, len(examples), chunksize):
            inputs = examples[i: i + chunksize]
            new_examples.append(
                (
                    inputs,
                    label_map,
                    max_seq_length,
                    tokenizer,
                    cls_token_at_end,
                    cls_token,
                    cls_token_segment_id,
                    sep_token,
                    sep_token_extra,
                    pad_on_left,
                    pad_token,
                    pad_token_segment_id,
                    pad_token_label_id,
                    sequence_a_segment_id,
                    mask_padding_with_zero,
                )
            )

        # with Pool(process_count) as p:
        pool = Pool(processes=process_count)
        features = list(
            tqdm(
                pool.imap(
                    convert_examples_with_multiprocessing,
                    new_examples,
                ),
                total=len(new_examples),
                disable=silent,
            )
        )

        features = [feature for feature_group in features for feature in
                        feature_group]
        pool.close()

    else:
        features = [
            convert_example_to_feature(
                example,
                label_map,
                max_seq_length,
                tokenizer,
                cls_token_at_end,
                cls_token,
                cls_token_segment_id,
                sep_token,
                sep_token_extra,
                pad_on_left,
                pad_token,
                pad_token_segment_id,
                pad_token_label_id,
                sequence_a_segment_id,
                mask_padding_with_zero,
            )
            for example in tqdm(examples, disable=silent)
        ]
    return features


def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return [
            "O",
            "B-MISC",
            "I-MISC",
            "B-PER",
            "I-PER",
            "B-ORG",
            "I-ORG",
            "B-LOC",
            "I-LOC",
        ]


def preprocess_batch_for_hf_dataset(
        data,
        label_list,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-1,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        silent=False,
):
    sequence_lengths = []
    all_words = []
    for seq in data["words"]:
        sequence_lengths.append(len(seq))
        all_words.extend(
            seq)  # Need to check whether adding the prefix space helps

    tokenized_word_ids_all = \
        tokenizer(text=all_words, add_special_tokens=False)[
            "input_ids"
        ]

    tokenized_word_ids_batch = []
    tokenized_word_ids_batch = [
        tokenized_word_ids_all[
        len(tokenized_word_ids_batch): len(tokenized_word_ids_batch) + seq_len
        ]
        for seq_len in sequence_lengths
    ]

    examples = [
        InputExample(guid, words, labels, tokenized_word_ids=tokenized_ids)
        for guid, words, labels, tokenized_ids in zip(
            data["sentence_id"], data["words"], data["labels"],
            tokenized_word_ids_batch
        )
    ]
    label_map = {label: i for i, label in enumerate(label_list)}

    features = [
        convert_example_to_feature(
            example,
            label_map=label_map,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            cls_token_at_end=cls_token_at_end,
            cls_token=cls_token,
            cls_token_segment_id=cls_token_segment_id,
            sep_token=sep_token,
            sep_token_extra=sep_token_extra,
            pad_on_left=pad_on_left,
            pad_token=pad_token,
            pad_token_segment_id=pad_token_segment_id,
            pad_token_label_id=pad_token_label_id,
            sequence_a_segment_id=sequence_a_segment_id,
            mask_padding_with_zero=mask_padding_with_zero,
            return_input_feature=False,
        )
        for example in tqdm(examples, disable=silent)
    ]

    feature_dict = {}
    feature_names = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "labels",
    ]

    for i, feature in enumerate(zip(*features)):
        feature_dict[feature_names[i]] = list(feature)

    return feature_dict


def load_hf_dataset(
        data,
        tokenizer,
        label_list,
        max_seq_length,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-1,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        silent=False,
        args=None,
):
    if isinstance(data, str):
        # dataset = load_dataset("conll2003", data_files=data)
        dataset = load_dataset(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "ner_dataset_loading_script"
            ),
            data_files=data,
            download_mode="force_redownload"
            if args.reprocess_input_data
            else "reuse_dataset_if_exists",
        )
    else:
        raise TypeError(
            "{} is not a path to a data file (e.g. tsv). The input must be a data file for NERModel.".format(
                data
            )
        )

    dataset = dataset.map(
        lambda x: preprocess_batch_for_hf_dataset(
            x,
            tokenizer,
            label_list,
            max_seq_length,
            cls_token_at_end=cls_token_at_end,
            cls_token=cls_token,
            cls_token_segment_id=cls_token_segment_id,
            sep_token=sep_token,
            sep_token_extra=sep_token_extra,
            pad_on_left=pad_on_left,
            pad_token=pad_token,
            pad_token_segment_id=pad_token_segment_id,
            pad_token_label_id=pad_token_label_id,
            sequence_a_segment_id=sequence_a_segment_id,
            mask_padding_with_zero=mask_padding_with_zero,
            silent=silent,
        ),
        batched=True,
    )

    dataset.set_format(
        type="pt",
        columns=["input_ids", "token_type_ids", "attention_mask", "labels"]
    )

    if isinstance(data, str):
        # This is not necessarily a train dataset. The datasets library insists on calling it train.
        return dataset["train"]
    else:
        return dataset


class LazyNERDataset(Dataset):
    def __init__(self, data_file, tokenizer, args):
        self.data_file = data_file
        self.lazy_loading_start_line = (
            args.lazy_loading_start_line if args.lazy_loading_start_line else 1
        )
        self.example_lines, self.num_entries = self._get_examples(
            self.data_file, self.lazy_loading_start_line
        )
        self.tokenizer = tokenizer
        self.args = args
        self.pad_token_label_id = CrossEntropyLoss().ignore_index

    @staticmethod
    def _get_examples(data_file, lazy_loading_start_line):
        example_lines = {}
        start = lazy_loading_start_line
        entry_num = 0
        with open(data_file, encoding="utf-8") as f:
            for line_idx, _ in enumerate(f, 1):
                if _ == "\n" and line_idx > lazy_loading_start_line:
                    example_lines[entry_num] = (start, line_idx)
                    start = line_idx + 1
                    entry_num += 1

        return example_lines, entry_num

    def __getitem__(self, idx):
        start, end = self.example_lines[idx]
        words, labels = [], []
        for idx in range(start, end):
            line = linecache.getline(self.data_file, idx).rstrip("\n")
            splits = line.split(" ")
            words.append(splits[0])
            if len(splits) > 1:
                labels.append(splits[-1].replace("\n", ""))
            else:
                # Examples could have no label for mode = "test"
                labels.append("O")

        example = InputExample(
            guid="%s-%d".format("train", idx), words=words, labels=labels
        )

        label_map = {label: i for i, label in enumerate(self.args.labels_list)}

        example_row = (
            example,
            label_map,
            self.args.max_seq_length,
            self.tokenizer,
            bool(self.args.model_type in ["xlnet"]),
            self.tokenizer.cls_token,
            2 if self.args.model_type in ["xlnet"] else 0,
            self.tokenizer.sep_token,
            bool(self.args.model_type in ["roberta"]),
            bool(self.args.model_type in ["xlnet"]),
            self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
            4 if self.args.model_type in ["xlnet"] else 0,
            self.pad_token_label_id,
            0,
            True,
        )

        features = convert_example_to_feature(*example_row)
        all_input_ids = torch.tensor(features.input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(features.input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(features.segment_ids, dtype=torch.long)
        all_label_ids = torch.tensor(features.label_ids, dtype=torch.long)
        return (all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    def __len__(self):
        return self.num_entries


def flatten_results(results, parent_key="", sep="/"):
    out = []
    if isinstance(results, Mapping):
        for key, value in results.items():
            pkey = parent_key + sep + str(key) if parent_key else str(key)
            out.extend(flatten_results(value, parent_key=pkey).items())
    elif isinstance(results, Iterable):
        for key, value in enumerate(results):
            pkey = parent_key + sep + str(key) if parent_key else str(key)
            out.extend(flatten_results(value, parent_key=pkey).items())
    else:
        out.append((parent_key, results))
    return dict(out)

import wandb
from dataclasses import asdict
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
def turn_span_df_2_token_list(data):
    if isinstance(data, list):
        data = pd.concat(data)
    pred_ids = []
    gold_ids = []
    pred_cls = []
    gold_cls = []
    for s, d in data.groupby('sent_id'):
        pred_id = d['predictions'].tolist()
        gold_id = d['gold_id'].tolist()
        p_cls = 'oth'
        g_cls = 'oth'
        if pred_id != ['O'] * len(pred_id):
            p_cls = [cl.split('-')[1] for cl in pred_id if cl != 'O'][0]
            pred_id = ['B-' + cl.split('-')[1] if 'I-' in cl else cl for cl in pred_id]
        if gold_id != ['O'] * len(gold_id):
            g_cls = [cl.split('-')[1] for cl in gold_id if cl != 'O'][0]
            gold_id = ['B-' + cl.split('-')[1] if 'I-' in cl else cl for cl in gold_id]

        gold_ids.append(gold_id)
        pred_ids.append(pred_id)
        gold_cls.append(g_cls)
        pred_cls.append(p_cls)
    return gold_ids, pred_ids, gold_cls, pred_cls

def calc_token_level_f1_for_df_data(data):
    gold_ids, pred_ids, gold_cls, pred_cls = turn_span_df_2_token_list(data)
    p = round(precision_score(gold_ids, pred_ids), 4)
    r = round(recall_score(gold_ids, pred_ids), 4)
    f1 = round(f1_score(gold_ids, pred_ids), 4)
    f0_5 = round(1.25*p*r/(0.25*p+r), 4)
    em = round(sum([1 for g, p in zip(pred_ids, gold_ids) if g == p])/len(pred_ids), 4)
    acc = round(accuracy_score(gold_cls, pred_cls), 4)
    print("p, r, f1, f0_5, em, acc")
    print(p, r, f1, f0_5, em, acc)
    return p, r, f1, f0_5, em, acc

# ------------------------------------------------------------
def calc_metrics(args,
                 eval_data,
                 model_outputs,
                 out_label_list,
                 preds_list,
                 eval_loss,
                 eval_output_dir,
                 wandb_log,
                 tgt_out_label_list=None,
                 tgt_preds_list=None,
                 src_out_label_list=None,
                 src_preds_list=None,
                 **kwargs):
    new_data = []
    for i, (sent_id, d) in enumerate(eval_data.groupby(['sentence_id'])):
        ogold_id = out_label_list[i]
        opred_id = preds_list[i]
        if '[MOD]' in d['words'].tolist():
            len_sent = d['words'].tolist().index('[MOD]')
        else:
            len_sent = len(d['words'].tolist())
        # if min(len(ogold_id), len(opred_id)) < len_sent:
        #     ogold_id.extend(['O'] * (len_sent - len(ogold_id)))
        #     opred_id.extend(['O'] * (len_sent - len(opred_id)))
        gold_id = ogold_id[:len_sent]
        pred_id = opred_id[:len_sent]
        # length = len(gold_id)

        if 'parsing_embedding' in eval_data.keys():
            for a, b, c, d, e in zip([sent_id] * len_sent, gold_id, d['parsing_embedding'].tolist()[: len_sent],
                                     pred_id, d['cls_labels'].tolist()[: len_sent]):
                if b in ['B-oth', 'I-oth']: b = 'O'
                if d in ['B-oth', 'I-oth']: d = 'O'
                new_data.append([a, b, c, d, e])
        else:
            for a, b, d, e in zip([sent_id] * len_sent, gold_id, pred_id, d['cls_labels'].tolist()[: len_sent]):
                if b in ['B-oth', 'I-oth']: b = 'O'
                if d in ['B-oth', 'I-oth']: d = 'O'
                new_data.append([a, b, d, e])
    if 'parsing_embedding' in eval_data.keys():
        new_df = pd.DataFrame(new_data)
        new_df.columns = ['sent_id', 'gold_id', 'parsing_embedding', 'predictions', 'cls_label']
    else:
        new_df = pd.DataFrame(new_data)
        new_df.columns = ['sent_id', 'gold_id', 'predictions', 'cls_label']

    p, r, f1, f0_5, em, acc = calc_token_level_f1_for_df_data(new_df)
    return {
        "eval_loss": eval_loss,
        "precision": p,
        "recall": r,
        "f1_score": f1,
        "f0_5_score": f0_5,
    }
    # extra_metrics = {}
    # for metric, func in kwargs.items():
    #     if metric.startswith("prob_"):
    #         extra_metrics[metric] = func(out_label_list, model_outputs)
    #     else:
    #         extra_metrics[metric] = func(out_label_list, preds_list)
    #
    # new_out_label_list = []
    # new_preds_list = []
    #
    # for o, p in zip(out_label_list, preds_list):
    #     if len(o) == len(p):
    #         new_out_label_list.append(o)
    #         new_preds_list.append(p)
    # if args.is_stbert or args.stbert_ensemble:
    #     def _filter_unequal(tgt_out_label_list, tgt_preds_list, pref):
    #         tmp_tgt_out_label_list, tmp_tgt_preds_list = [], []
    #         for o, p in zip(tgt_out_label_list, tgt_preds_list):
    #             if len(o) == len(p):
    #                 tmp_tgt_out_label_list.append(o)
    #                 tmp_tgt_preds_list.append(p)
    #         extra_metrics[pref + '_f1'] = f1_score(tmp_tgt_out_label_list, tmp_tgt_preds_list)
    #         extra_metrics[pref + 'recall'] = recall_score(tmp_tgt_out_label_list, tmp_tgt_preds_list)
    #         extra_metrics[pref + 'precision'] = precision_score(tmp_tgt_out_label_list, tmp_tgt_preds_list)
    #         return extra_metrics
    #
    #     extra_metrics.update(_filter_unequal(tgt_out_label_list, tgt_preds_list, pref='tgt'))
    #     extra_metrics.update(_filter_unequal(src_out_label_list, src_preds_list, pref='tgt'))
    #
    # print('len origin data:{}, len new data: {}'.format(len(out_label_list), len(new_preds_list)))
    #
    # result = {
    #     "eval_loss": eval_loss,
    #     "precision": precision_score(new_out_label_list, new_preds_list),
    #     "recall": recall_score(new_out_label_list, new_preds_list),
    #     "f1_score": f1_score(new_out_label_list, new_preds_list),
    #     **extra_metrics,
    # }
    # print(result)
    #
    # os.makedirs(eval_output_dir, exist_ok=True)
    # output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    # with open(output_eval_file, "w") as writer:
    #     if args.classification_report:
    #         cls_report = classification_report(out_label_list, preds_list, digits=4)
    #         writer.write("{}\n".format(cls_report))
    #     for key in sorted(result.keys()):
    #         writer.write("{} = {}\n".format(key, str(result[key])))
    #
    # if args.wandb_project and wandb_log:
    #     wandb.init(
    #         project=args.wandb_project,
    #         config={**asdict(args)},
    #         **args.wandb_kwargs,
    #     )
    #     wandb.run._label(repo="simpletransformers")
    #
    #     labels_list = sorted(args.labels_list)
    #
    #     truth = [tag for out in out_label_list for tag in out]
    #     preds = [tag for pred_out in preds_list for tag in pred_out]
    #     outputs = [
    #         np.mean(logits, axis=0) for output in model_outputs for logits in output
    #     ]
    #
    #     # ROC
    #     wandb.log({"roc": wandb.plots.ROC(truth, outputs, labels_list)})
    #
    #     # Precision Recall
    #     wandb.log({"pr": wandb.plots.precision_recall(truth, outputs, labels_list)})
    #
    #     # Confusion Matrix
    #     wandb.sklearn.plot_confusion_matrix(
    #         truth,
    #         preds,
    #         labels=labels_list,
    #     )
