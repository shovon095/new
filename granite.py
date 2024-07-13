import logging
import os
import torch
import datasets
import transformers
from transformers import (
    HfArgumentParser,
    set_seed,
    EarlyStoppingCallback,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
from transformers.trainer_utils import get_last_checkpoint
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional
import utils.tool
from utils.dataset_bird import TokenizedDataset
from utils.trainer import EvaluateFriendlySeq2SeqTrainer
from utils.training_arguments import WrappedSeq2SeqTrainingArguments
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    train_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    test_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the test data."}
    )

def main():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    logging.basicConfig(level=logging.INFO)

    parser = HfArgumentParser((WrappedSeq2SeqTrainingArguments, DataTrainingArguments))
    parser.add_argument("--granite_model_path", type=str, required=True, help="Path to the Granite model.")
    parser.add_argument("--data_store_path", type=str, required=True, help="Path to the dataset storage.")
    parser.add_argument("--seq2seq_constructor", type=str, required=True, help="Seq2Seq constructor name.")
    parser.add_argument("--evaluate_tool", type=str, required=True, help="Evaluation tool name.")
    parser.add_argument("--bert_location", type=str, required=True, help="BERT model location.")
    training_args, data_args, other_args = parser.parse_args_into_dataclasses()
    
    set_seed(training_args.seed)

    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the --output_dir or add --overwrite_output_dir to train from scratch."
            )

    os.makedirs(training_args.output_dir, exist_ok=True)

    cache_root = os.path.join('output', 'cache')
    os.makedirs(cache_root, exist_ok=True)
    raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(path=data_args.train_data_path,
                                                                     cache_dir=other_args.data_store_path)
    seq2seq_dataset_split: tuple = utils.tool.get_constructor(other_args.seq2seq_constructor)(other_args).to_seq2seq(
        raw_datasets_split, cache_root)

    evaluator = utils.tool.get_evaluator(other_args.evaluate_tool)(other_args)

    model = AutoModelForSeq2SeqLM.from_pretrained(training_args.granite_model_path).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    tokenizer = AutoTokenizer.from_pretrained(training_args.granite_model_path)

    seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = None, None, None
    if len(seq2seq_dataset_split) == 2:
        seq2seq_train_dataset, seq2seq_eval_dataset = seq2seq_dataset_split
    elif len(seq2seq_dataset_split) == 3:
        seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = seq2seq_dataset_split
    else:
        raise ValueError("Other split not supported yet.")

    train_dataset = TokenizedDataset(other_args, training_args, tokenizer,
                                     seq2seq_train_dataset) if seq2seq_train_dataset else None
    eval_dataset = TokenizedDataset(other_args, training_args, tokenizer,
                                    seq2seq_eval_dataset) if seq2seq_eval_dataset else None

    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5)
    trainer = EvaluateFriendlySeq2SeqTrainer(
        args=training_args,
        model=model,
        evaluator=evaluator,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=seq2seq_eval_dataset,
        callbacks=[early_stopping_callback],
    )
    print('Trainer built successfully.')

    if training_args.load_weights_from:
        state_dict = torch.load(os.path.join(training_args.load_weights_from, transformers.WEIGHTS_NAME), map_location="cpu")
        trainer.model.load_state_dict(state_dict, strict=True)
        del state_dict

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = len(train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            test_dataset=seq2seq_test_dataset if seq2seq_test_dataset else eval_dataset,
            test_examples=seq2seq_test_dataset if seq2seq_test_dataset else seq2seq_eval_dataset,
            metric_key_prefix="predict"
        )
        metrics = predict_results.metrics
        max_predict_samples = len(seq2seq_test_dataset if seq2seq_test_dataset else eval_dataset)
        metrics["predict_samples"] = min(max_predict_samples, len(seq2seq_test_dataset if seq2seq_test_dataset else eval_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions to a file
        predictions = predict_results.predictions
        output_prediction_file = os.path.join(training_args.output_dir, "predictions.txt")
        with open(output_prediction_file, "w") as writer:
            for prediction in predictions:
                writer.write(f"{prediction}\n")

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = HfArgumentParser((WrappedSeq2SeqTrainingArguments, DataTrainingArguments))
    parser.add_argument("--granite_model_path", type=str, required=True, help="Path to the Granite model.")
    parser.add_argument("--data_store_path", type=str, required=True, help="Path to the dataset storage.")
    parser.add_argument("--seq2seq_constructor", type=str, required=True, help="Seq2Seq constructor name.")
    parser.add_argument("--evaluate_tool", type=str, required=True, help="Evaluation tool name.")
    parser.add_argument("--bert_location", type=str, required=True, help="BERT model location.")
    training_args, data_args, other_args = parser.parse_args_into_dataclasses()

    main()
