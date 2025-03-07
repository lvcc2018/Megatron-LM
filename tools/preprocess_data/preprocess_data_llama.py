# coding=utf-8
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Processing data for pretraining."""

import argparse
import json
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import torch

try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""


class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)

        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.", flush=True)
                exit(-1)

            library = "tokenizers/punkt/{}.pickle".format(self.args.lang)
            print("loading: " + library, flush=True)
            splitter = nltk.load(library)

            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text=splitter._params,
                    lang_vars=CustomLanguageVars()
                )
            else:
                Encoder.splitter = splitter
        else:
            Encoder.splitter = IdentitySplitter()

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}

        for key in self.args.json_keys:
            text = data[key]
            doc_ids = []

            for sentence in Encoder.splitter.tokenize(text):
                sentence_ids = Encoder.tokenizer.tokenize(sentence, bos=True, eos=True)
                if len(sentence_ids) > 0:
                    doc_ids.append(sentence_ids)

            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eod)

            ids[key] = doc_ids

        return ids, len(json_line)


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument("--input", type=str, required=True,
                       help="Path to input JSON")
    group.add_argument("--json-keys", nargs="+", default=["text"],
                       help="space separate listed of keys to extract from json")
    group.add_argument("--split-sentences", action="store_true",
                       help="Split documents into sentences.")
    group.add_argument("--keep-newlines", action="store_true",
                       help="Keep newlines between sentences when splitting.")

    group = parser.add_argument_group(title="tokenizer")
    group.add_argument("--tokenizer-type", type=str, required=True,
                       default="GPTSentencePieceTokenizer",
                       help="What type of tokenizer to use.")
    group.add_argument("--tokenizer-model",
                       help="Path to the tokenizer model")
    group.add_argument("--tokenizer-file",
                       help="Path to the tokenizer file")
    group.add_argument("--append-eod", action="store_true",
                       help="Append an eod_token to the end of a document.")
    group.add_argument("--lang", type=str, default="english",
                       help="Language to use for NLTK-powered sentence splitting.")


    group = parser.add_argument_group(title="output data")
    group.add_argument("--output-prefix", type=str, required=True,
                       help="Path to binary output file without suffix")
    group.add_argument("--dataset-impl", type=str, default="mmap",
                       choices=["lazy", "cached", "mmap"])

    group = parser.add_argument_group(title="runtime")
    group.add_argument("--workers", type=int, required=True,
                       help="Number of worker processes to launch")
    group.add_argument("--chunk-size", type=int, required=True,
                       help="Chunk size assigned to each worker process")
    group.add_argument("--log-interval", type=int, default=100,
                       help="Interval between progress updates")
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith("bert"):
        if not args.split_sentences:
            print("Bert tokenizer detected, \
                are you sure you don't want to split sentences?", flush=True)

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 4
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def main():
    args = get_args()
    startup_start = time.time()

    print(f"Opening {args.input}", flush=True)

    fd = open(args.input, "rb")

    if nltk_available and args.split_sentences:
        nltk.download("punkt", quiet=True)

    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_docs = pool.imap(encoder.encode, fd, args.chunk_size)

    if args.split_sentences:
        level = "sentence"
    else:
        level = "document"

    print(f"Vocab size: {tokenizer.vocab_size}", flush=True)
    print(f"Output prefix: {args.output_prefix}", flush=True)

    output_bin_files = {}
    output_idx_files = {}
    builders = {}

    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                      key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                      key, level)
        builders[key] = indexed_dataset.make_builder(
            output_bin_files[key],
            impl=args.dataset_impl,
            vocab_size=tokenizer.vocab_size
        )

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0

    print(f"Time to startup: {startup_end - startup_start}", flush=True)

    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed

        for key, sentences in doc.items():
            if len(sentences) == 0:
                continue

            for sentence in sentences:
                builders[key].add_item(torch.IntTensor(sentence))

            builders[key].end_document()

        if (i % args.log_interval) == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {i} documents",
                  f"({i / elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr,
                  flush=True)

    print("Done! Now finalizing.", flush=True)

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])


if __name__ == "__main__":
    main()
