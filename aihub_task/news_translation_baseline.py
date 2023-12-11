import argparse
import json
import os
import glob
import copy
import unicodedata
from shutil import copyfile
import random
from collections import Counter
import string
import re
import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    BertConfig,

    BertForSequenceClassification,
    DistilBertConfig,
    DistilBertForSequenceClassification,
)
import logging
logger = logging.getLogger(__name__)

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

CONFIG_CLASSES = {
    "kobert": BertConfig,
    "distilkobert": DistilBertConfig,
}

MODEL_FOR_SEQUENCE_CLASSIFICATION = {
    "kobert": BertForSequenceClassification,
    "distilkobert": DistilBertForSequenceClassification,
}

def train(args,
          model,
          train_dataset,):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.num_workers,)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    """
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    """
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0.0

    model.zero_grad()

    for epoch in range(int(args.num_train_epochs)):
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
            if args.model_type not in ["distilkobert", "xlm-roberta"]:
                inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids
            outputs = model(**inputs)

            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    len(train_dataloader) <= args.gradient_accumulation_steps
                    and (step + 1) == len(train_dataloader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        logger.info("Epoch %d done.", epoch+1)

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )

    model_to_save.save_pretrained(output_dir)

    logger.info("Saving model checkpoint to {}".format(output_dir))

    return global_step, tr_loss / global_step

def evaluate(args, model, eval_dataset, mode, global_step=None):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers,)

    if global_step != None:
        logger.info("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logger.info("***** Running evaluation on {} dataset *****".format(mode))
    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
    preds = None

    for batch in tqdm(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
            }
            if args.model_type not in ["distilkobert", "xlm-roberta"]:
                inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids

            outputs = model(**inputs)

            logits = outputs['logits']

        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)

    torch.save(torch.tensor(preds), os.path.join('./y_pred.pt'))


def main(args):
    # print("Training/evaluation parameters {}".format(args))

    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

    set_seed(args)

    train_dataset = [torch.load(f'/content/{args.train_file[0]}'), torch.load(f'/content/{args.train_file[1]}')] if args.train_file else None

    if train_dataset:
        train_dataset = train_dataset[0] + (train_dataset[1],)
        train_dataset = torch.utils.data.TensorDataset(*train_dataset)

    test_dataset = [torch.load(f'/content/{args.test_file[0]}')] if args.test_file else None

    if test_dataset:
        test_dataset = test_dataset[0]
        test_dataset = torch.utils.data.TensorDataset(*test_dataset)

    labels = [str(l) for l in train_dataset[:][-1].unique().tolist()]

    config = CONFIG_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
        num_labels=len(labels),
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
    )

    model = MODEL_FOR_SEQUENCE_CLASSIFICATION[args.model_type].from_pretrained(
        args.model_name_or_path,
        config=config
    )

    print("\n=================================")
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    print('Num Parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    os.remove('temp.p')
    print("=================================\n\n")

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)

    if args.do_train:
        global_step, tr_loss = train(args, model, train_dataset)
        logger.info(" global_step = {}, average loss = {}".format(global_step, tr_loss))

    if args.do_eval:
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True))
        )

        checkpoints = checkpoints[-1:]

        logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = MODEL_FOR_SEQUENCE_CLASSIFICATION[args.model_type].from_pretrained(checkpoint)
            model.to(args.device)
            evaluate(args, model, test_dataset, mode="test", global_step=global_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="yesno")
    parser.add_argument("--train_file", type=list, default=["datasets/train_x.pt", "datasets/train_y.pt"])
    parser.add_argument("--test_file", type=list, default=["datasets/test_x.pt"])
    parser.add_argument("--evaluate_test_during_training", type=bool, default=True)
    parser.add_argument("--eval_all_checkpoints", type=bool, default=True)
    parser.add_argument("--save_optimizer", type=bool, default=False)
    parser.add_argument("--do_lower_case", type=bool, default=False)
    parser.add_argument("--do_train", type=bool, default=True)
    parser.add_argument("--do_eval", type=bool, default=True)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--no_cuda", type=bool, default=False)
    parser.add_argument("--model_type", type=str, default="distilkobert")
    parser.add_argument("--model_name_or_path", type=str, default="monologg/distilkobert")
    parser.add_argument("--output_dir", type=str, default="distilkobert")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--logging_steps", type=int, default=2000)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("-f")

    args = parser.parse_args()

    main(args)
            