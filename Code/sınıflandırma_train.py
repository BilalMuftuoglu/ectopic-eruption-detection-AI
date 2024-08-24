from huggingface_hub import notebook_login,login
from transformers.utils import send_example_telemetry
from datasets import load_dataset
from datasets import load_metric
from transformers import DeiTForImageClassificationWithTeacher, AutoImageProcessor
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer,DeiTForImageClassification
import torch
import numpy as np
import subprocess

from torchvision.transforms import (
        CenterCrop,
        Compose,
        Normalize,
        RandomHorizontalFlip,
        RandomResizedCrop,
        Resize,
        ToTensor,
    )

login("YOUR_HUGGINGFACE_TOKEN")

model_checkpoint = "facebook/deit-base-distilled-patch16-224"
#model_checkpoint = "microsoft/beit-base-patch16-224" 
batch_size = 32 # batch size for training and evaluation

#pip install -q datasets transformers accelerate
# %%capture
# !sudo apt -qq install git-lfs
# !git config --global credential.helper store

subprocess.run(['pip', 'install', '-q', 'datasets', 'transformers', 'accelerate'])
subprocess.run(['sudo', 'apt', '-qq', 'install', 'git-lfs'])
subprocess.run(['git', 'config', '--global', 'credential.helper', 'store'])

send_example_telemetry("image_classification_notebook", framework="pytorch")

metric = load_metric("accuracy")

tooths = ["55","65","75","85"]

for tooth in tooths:
  for f in range(5):
    dataset = load_dataset(f"dataset/k-fold-all/{tooth}/fold{f+1}/train", data_dir="")

    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    image_processor  = AutoImageProcessor.from_pretrained(model_checkpoint)
    image_processor.size = {"height": 224, "width": 224}

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
        max_size = None
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)
        max_size = image_processor.size.get("longest_edge")

    print(size, crop_size, max_size)
    print(image_processor.size)
    print(image_processor)

    train_transforms = Compose(
            [
                RandomResizedCrop(crop_size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )

    val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(crop_size),
                ToTensor(),
                normalize,
            ]
        )

    def preprocess_train(example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch

    def preprocess_val(example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    splits = dataset["train"].train_test_split(test_size=0.15)
    train_ds = splits['train']
    val_ds = splits['test']

    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val)

    

    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    )

    #model = DeiTForImageClassification.from_pretrained(
     #   model_checkpoint,
      #  label2id=label2id,
       # id2label=id2label,
        #ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    #)

    model_name = model_checkpoint.split("/")[-1]
    args = TrainingArguments(
        f"{model_name}-hasta-{tooth}-fold{f+1}",
        remove_unused_columns=False,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=100,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=True,
    )

    # the compute_metrics function takes a Named Tuple as input:
    # predictions, which are the logits of the model as Numpy arrays,
    # and label_ids, which are the ground-truth labels as Numpy arrays.
    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    train_results = trainer.train()
    # rest is optional but nice to have
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate()
    # some nice to haves:
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_state()

    trainer.push_to_hub()

    #remove local log files
    #rm -rf {model_name}-hasta-{tooth}-fold{f+1}
