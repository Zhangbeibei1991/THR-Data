import sys

sys.path.append("...")
import os, time, functools
from t5.data import dataset_providers
from t5.data import preprocessors
import CHECKPOINT.src.configs.ee_t5 as t5_base
from CHECKPOINT.src.utils.rouge_utils import rouge_top_beam
from CHECKPOINT.src.utils.t5x_utils.test import test
import tensorflow as tf
import pandas as pd
from copy import deepcopy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
fine_tuning_cfg = t5_base.get_config()
fine_tuning_cfg.beam_size = 1

start = time.time()

data_dir = "sentence"

data_names = os.listdir(data_dir)
data_names = data_names

for item in data_names:
    if "-has" not in item:
        os.remove(os.path.join(data_dir, item))

pred_names = [item for item in data_names if "-Event.tsv" in item]
orig_names = [item for item in data_names if "-Event.tsv" not in item and "-has" in item and "-Entity" not in item]

for data_name in orig_names:
    data_name = data_name.split(".tsv")[0]
    save_name = os.path.join(data_dir, data_name + "-Event.tsv")
    if data_name + "-Event.tsv" in pred_names:
        print(f"{data_name} has been processed, next!")
        continue
    else:
        print(f"precessing: {data_name}")
    data = pd.read_csv(os.path.join(data_dir, data_name + ".tsv"), sep="\t")
    data_copy = deepcopy(data)
    del data["info"]
    dataset = data_name.split("-has")[0]
    data.to_csv(os.path.join(data_dir, dataset + ".tsv"), sep="\t", index=0)
    train_file = "train" + ".tsv"
    test_file = dataset + ".tsv"
    TaskRegistry = dataset_providers.TaskRegistry
    TaskRegistry.remove("event_extraction_task")

    TaskRegistry.add(
        "event_extraction_task",
        dataset_providers.TextLineTask,
        split_to_filepattern={
            "train": os.path.join("../CHECKPOINT/data/datasets/biot2e", train_file),
            "validation": os.path.join(data_dir, test_file)
        },
        skip_header_lines=1,
        text_preprocessor=preprocessors.preprocess_tsv,
        metric_fns=[functools.partial(
            # rouge_top_beam, beam_size=fine_tuning_cfg.beam_size)]
            rouge_top_beam, beam_size=1)]
    )
    test(task_name="event_extraction_task", model_dir="../CHECKPOINT/data/model_data/ee/t5x/best_checkpoint/",
         data_dir=os.path.join(data_dir, test_file),
         config=fine_tuning_cfg, output_prediction_postfix=dataset, ee=True)

    answer = []
    with open(f"./preds_EE_{dataset}.txt", encoding="utf-8", mode="r") as f:
        for s, line in enumerate(f.readlines()):
            if len(line.strip()) != 0:
                data_copy["text_label"][s] = line.strip()

    data_copy.to_csv(os.path.join(data_dir, data_name + "-Event.tsv"), sep="\t", index=0)
    os.remove(os.path.join(data_dir, dataset + ".tsv"))
    os.remove(f"./preds_EE_{dataset}.txt")
