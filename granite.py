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
import utils.tool
from utils.configue import Configure
from utils.dataset_bird import TokenizedDataset
from utils.trainer import EvaluateFriendlySeq2SeqTrainer
from utils.training_arguments import WrappedSeq2SeqTrainingArguments
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

logger = logging.getLogger(__name__)

def main():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    logging.basicConfig(level=logging.INFO)

    parser = HfArgumentParser((WrappedSeq2SeqTrainingArguments,))
    parser.add_argument("--granite_model_path", type=str, required=True, help="Path to the Granite model.")
    training_args, other_args = parser.parse_args_into_dataclasses()
    
    if not training_args.cfg:
        training_args.cfg = "default_config.yaml"
    
    print(f"training_args.cfg: {training_args.cfg}")
    
    set_seed(training_args.seed)
    args = Configure.Get(training_args.cfg)

    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if 'checkpoint-???' in args.bert.location:
        args.bert.location = get_last_checkpoint(
            os.path.dirname(args.bert.location.model_name_or_path))
        logger.info(f"Resolve model_name_or_path to {args.bert.location.model_name_or_path}")

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

    if not args.arg_paths:
        cache_root = os.path.join('output', 'cache')
        os.makedirs(cache_root, exist_ok=True)
        raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(path=training_args.train_data_path,
                                                                         cache_dir=args.dataset.data_store_path)
        seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).to_seq2seq(
            raw_datasets_split, cache_root)
    else:
        cache_root = os.path.join('output', 'cache')
        os.makedirs(cache_root, exist_ok=True)
        meta_tuning_data = {}
        for task, arg_path in args.arg_paths:
            task_args = Configure.Get(arg_path)
            task_args.bert = args.bert
            print('task_args.bert.location:', task_args.bert.location)
            task_raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(
                path=task_args.dataset.loader_path,
                cache_dir=task_args.dataset.data_store_path)
            task_seq2seq_dataset_split: tuple = utils.tool.get_constructor(task_args.seq2seq.constructor)(task_args).\
                to_seq2seq(task_raw_datasets_split, cache_root)

            meta_tuning_data[arg_path] = task_seq2seq_dataset_split

        seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).\
            to_seq2seq(meta_tuning_data)

    evaluator = utils.tool.get_evaluator(args.evaluate.tool)(args)

    model = AutoModelForSeq2SeqLM.from_pretrained(training_args.granite_model_path).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    tokenizer = AutoTokenizer.from_pretrained(training_args.granite_model_path)

    seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = None, None, None
    if len(seq2seq_dataset_split) == 2:
        seq2seq_train_dataset, seq2seq_eval_dataset = seq2seq_dataset_split
    elif len(seq2seq_dataset_split) == 3:
        seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = seq2seq_dataset_split
    else:
        raise ValueError("Other split not support yet.")

    train_dataset = TokenizedDataset(args, training_args, tokenizer,
                                     seq2seq_train_dataset) if seq2seq_train_dataset else None
    eval_dataset = TokenizedDataset(args, training_args, tokenizer,
                                    seq2seq_eval_dataset) if seq2seq_eval_dataset else None

    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=args.seq2seq.patience if args.seq2seq.patience else 5)
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
    print('Trainer build successfully.')

    if training_args.load_weights_from:
        state_dict = torch.load(os.path.join(training_args.load_weights_from, transformers.WEIGHTS_NAME), map_location="cpu")
        trainer.model.load_state_dict(state_dict, strict=True)
        del state_dict

    if args.load_multiple_prefix_module_weights_from:
        reconstruct_state_dict = OrderedDict()
        for task_name, module_weight_location in args.load_multiple_prefix_module_weights_from:
            state_dict = torch.load(os.path.join(module_weight_location, transformers.WEIGHTS_NAME), map_location="cpu")
            MULTI_PREFIX_ATTR_NAME = "multi_prefix"
            for weight_name, stored_tensor in state_dict.items():
                if str(weight_name).startswith("pretrain_model"):
                    continue
                reconstruct_state_dict['{}.{}.{}'.format(MULTI_PREFIX_ATTR_NAME, "_".join(task_name.split("_")[:-1]), weight_name)] = stored_tensor
        trainer.model.load_state_dict(reconstruct_state_dict, strict=False)
        del reconstruct_state_dict

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

def classwrapper(sql, question_id, db_schema):
    sql=trainer,evaluate 

if __name__ == "__main__":
    parser = HfArgumentParser((WrappedSeq2SeqTrainingArguments,))
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test data.")
    parser.add_argument("--granite_model_path", type=str, required=True, help="Path to the Granite model.")
    training_args, other_args = parser.parse_args_into_dataclasses()

    main()
