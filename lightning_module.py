"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import math
import random
import re
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from nltk import edit_distance
from pytorch_lightning.utilities import rank_zero_only
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from donut import DonutConfig, DonutModel, DonutClipModel, DonutClipConfig
import teds


class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.validation_metric == "teds":
            self.teds = teds.TEDS()
        elif config.validation_metric == "teds_structure":
            self.teds = teds.TEDS(True)
        self.validation_metric = config.validation_metric

        task_start_tokens = []
        prompt_end_tokens = []
        for i, dataset_name_or_path in enumerate(config.dataset_name_or_paths):
            if config.get("task_start_tokens", None):
                task_start_token = config.task_start_tokens[0]
            else:
                task_start_token = f"<s_{config.task_name}>"
            if "docvqa" in dataset_name_or_path:
                prompt_end_token = "<s_answer>"
            else:
                prompt_end_token = f"<s_{config.task_name}>"
            task_start_tokens.append(task_start_token)
            prompt_end_tokens.append(prompt_end_token)

        if self.config.get("pretrained_model_name_or_path", False):
            self.model = DonutModel.from_pretrained(
                self.config.pretrained_model_name_or_path,
                input_size=self.config.input_size,
                max_length=self.config.max_length,
                align_long_axis=self.config.align_long_axis,
                ignore_mismatched_sizes=True,
                use_fast_tokenizer=self.config.use_fast_tokenizer,
                tokenizer_name_or_path=self.config.tokenizer_name_or_path,
                vision_model_name=self.config.vision_model_name,
                bart_prtrained_path=self.config.bart_prtrained_path,
                special_tokens=task_start_tokens + prompt_end_tokens,
                swin_pretrained_path=self.config.swin_pretrained_path,
                window_size=self.config.window_size,
                swin_model_size=self.config.swin_model_size
            )
        else:
            self.model = DonutModel(
                config=DonutConfig(
                    input_size=self.config.input_size,
                    max_length=self.config.max_length,
                    align_long_axis=self.config.align_long_axis,
                    tokenizer_name_or_path=self.config.tokenizer_name_or_path,
                    use_fast_tokenizer=self.config.use_fast_tokenizer,
                    vision_model_name=self.config.vision_model_name,
                    bart_prtrained_path=self.config.bart_prtrained_path,
                    special_tokens=task_start_tokens + prompt_end_tokens,
                    swin_pretrained_path=self.config.swin_pretrained_path,
                    window_size=self.config.window_size,
                    swin_model_size=self.config.swin_model_size
                    # with DonutConfig, the architecture customization is available, e.g.,
                    # encoder_layer=[2,2,14,2], decoder_layer=4, ...
                )
            )

    def training_step(self, batch, batch_idx):
        image_tensors, decoder_input_ids, decoder_labels = list(), list(), list()
        for batch_data in batch:
            image_tensors.append(batch_data[0])
            decoder_input_ids.append(batch_data[1][:, :-1])
            decoder_labels.append(batch_data[2][:, 1:])
        image_tensors = torch.cat(image_tensors)
        decoder_input_ids = torch.cat(decoder_input_ids)
        decoder_labels = torch.cat(decoder_labels)
        loss = self.model(image_tensors, decoder_input_ids, decoder_labels)[0]
        self.log_dict({"train_loss": loss}, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        image_tensors, decoder_input_ids, prompt_end_idxs, answers = batch
        decoder_prompts = pad_sequence(
            [input_id[: end_idx + 1] for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)],
            batch_first=True,
        )

        preds = self.model.inference(
            image_tensors=image_tensors,
            prompt_tensors=decoder_prompts,
            return_json=False,
            return_attentions=False,
        )["predictions"]

        scores = list()
        for pred, answer in zip(preds, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            answer = re.sub(r"<.*?>", "", answer, count=1)
            answer = answer.replace(self.model.decoder.tokenizer.eos_token, "")
            if self.validation_metric.startswith("teds"):
                pred = teds.postprocess_html_tag(pred)
                answer = teds.postprocess_html_tag(answer)
                scores.append(self.teds.evaluate(pred, answer))
            else:
                scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                self.print(f"Prediction: {pred}")
                self.print(f"    Answer: {answer}")
                if self.validation_metric.startswith("teds"):
                    self.print(f" TEDS: {scores[0]}")
                else:
                    self.print(f" Normed ED: {scores[0]}")

        return scores

    def validation_epoch_end(self, validation_step_outputs):
        num_of_loaders = len(self.config.dataset_name_or_paths)
        if num_of_loaders == 1:
            validation_step_outputs = [validation_step_outputs]
        assert len(validation_step_outputs) == num_of_loaders
        cnt = [0] * num_of_loaders
        total_metric = [0] * num_of_loaders
        val_metric = [0] * num_of_loaders
        for i, results in enumerate(validation_step_outputs):
            for scores in results:
                cnt[i] += len(scores)
                total_metric[i] += np.sum(scores)
            val_metric[i] = total_metric[i] / cnt[i]
            val_metric_name = f"val_metric_{i}th_dataset"
            self.log_dict({val_metric_name: val_metric[i]}, sync_dist=True)
        self.log_dict({"val_metric": np.sum(total_metric) / np.sum(cnt)},
                      sync_dist=True)
        print("average val teds", np.sum(total_metric) / np.sum(cnt))

    def configure_optimizers(self):

        max_iter = None

        if self.config.get("max_epochs", None):
            assert len(self.config.train_batch_sizes) == 1, "Set max_epochs only if the number of datasets is 1"
            max_iter = (self.config.max_epochs * self.config.num_training_samples_per_epoch) / (
                    self.config.train_batch_sizes[0] * torch.cuda.device_count() * self.config.get("num_nodes", 1)
            )

        if self.config.get("max_steps", None):
            max_iter = min(self.config.max_steps, max_iter) if max_iter is not None else self.config.max_steps

        assert max_iter is not None
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, self.config.warmup_steps),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items["exp_name"] = f"{self.config.get('exp_name', '')}"
        items["exp_version"] = f"{self.config.get('exp_version', '')}"
        return items

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        save_path = Path(self.config.result_path) / self.config.exp_name / self.config.exp_version
        self.model.save_pretrained(save_path)
        self.model.decoder.tokenizer.save_pretrained(save_path)


class DonutDataPLModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_batch_sizes = self.config.train_batch_sizes
        self.val_batch_sizes = self.config.val_batch_sizes
        self.train_datasets = []
        self.val_datasets = []
        self.g = torch.Generator()
        self.g.manual_seed(self.config.seed)

    def train_dataloader(self):
        loaders = list()
        for train_dataset, batch_size in zip(self.train_datasets, self.train_batch_sizes):
            loaders.append(
                DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    num_workers=self.config.num_workers,
                    pin_memory=True,
                    worker_init_fn=self.seed_worker,
                    generator=self.g,
                    shuffle=True,
                )
            )
        return loaders

    def val_dataloader(self):
        loaders = list()
        for val_dataset, batch_size in zip(self.val_datasets, self.val_batch_sizes):
            loaders.append(
                DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    pin_memory=True,
                    shuffle=False,
                )
            )
        return loaders

    @staticmethod
    def seed_worker(wordker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


class DonutClipModelPLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.validation_metric == "teds":
            self.teds = teds.TEDS()
        elif config.validation_metric == "teds_structure":
            self.teds = teds.TEDS(True)
        self.validation_metric = config.validation_metric

        task_start_tokens = []
        prompt_end_tokens = []
        for i, dataset_name_or_path in enumerate(config.dataset_name_or_paths):
            if config.get("task_start_tokens", None):
                task_start_token = config.task_start_tokens[0]
            else:
                task_start_token = f"<s_{config.task_name}>"
            if "docvqa" in dataset_name_or_path:
                prompt_end_token = "<s_answer>"
            else:
                prompt_end_token = f"<s_{config.task_name}>"
            task_start_tokens.append(task_start_token)
            prompt_end_tokens.append(prompt_end_token)

        if self.config.get("pretrained_model_name_or_path", False):
            self.model = DonutClipModel.from_pretrained(
                self.config.pretrained_model_name_or_path,
                input_size=self.config.input_size,
                max_length=self.config.max_length,
                align_long_axis=self.config.align_long_axis,
                ignore_mismatched_sizes=True,
                use_fast_tokenizer=self.config.use_fast_tokenizer,
                tokenizer_name_or_path=self.config.tokenizer_name_or_path,
                vision_model_name=self.config.vision_model_name,
                bart_prtrained_path=self.config.bart_prtrained_path,
                special_tokens=task_start_tokens + prompt_end_tokens,
                swin_pretrained_path=self.config.swin_pretrained_path,
                window_size=self.config.window_size,
                swin_model_size=self.config.swin_model_size
            )
        else:
            self.model = DonutClipModel(
                config=DonutClipConfig(
                    input_size=self.config.input_size,
                    max_length=self.config.max_length,
                    align_long_axis=self.config.align_long_axis,
                    tokenizer_name_or_path=self.config.tokenizer_name_or_path,
                    use_fast_tokenizer=self.config.use_fast_tokenizer,
                    vision_model_name=self.config.vision_model_name,
                    bart_prtrained_path=self.config.bart_prtrained_path,
                    special_tokens=task_start_tokens + prompt_end_tokens,
                    swin_pretrained_path=self.config.swin_pretrained_path,
                    window_size=self.config.window_size,
                    swin_model_size=self.config.swin_model_size
                    # with DonutConfig, the architecture customization is available, e.g.,
                    # encoder_layer=[2,2,14,2], decoder_layer=4, ...
                )
            )

    def training_step(self, batch, batch_idx):
        image_tensors, decoder_input_ids = list(), list()
        for batch_data in batch:
            image_tensors.append(batch_data[0])
            decoder_input_ids.append(batch_data[1][:, :-1])
        image_tensors = torch.cat(image_tensors)
        decoder_input_ids = torch.cat(decoder_input_ids)
        print(self.model(image_tensors, decoder_input_ids, return_loss=True, return_dict=True))
        loss = self.model(image_tensors, decoder_input_ids, return_loss=True, return_dict=True)[0]
        self.log_dict({"train_loss": loss}, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        image_tensors, decoder_input_ids, prompt_end_idxs, answers = batch
        decoder_prompts = pad_sequence(
            [input_id[: end_idx + 1] for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)],
            batch_first=True,
        )

        loss = self.model(image_tensors, decoder_input_ids, return_loss=True, return_dict=True)[0]
        print("loss", loss)

        return [loss, len(image_tensors)]

    def validation_epoch_end(self, validation_step_outputs):
        num_of_loaders = len(self.config.dataset_name_or_paths)
        if num_of_loaders == 1:
            validation_step_outputs = [validation_step_outputs]
        assert len(validation_step_outputs) == num_of_loaders
        cnt = [0] * num_of_loaders
        total_metric = [0] * num_of_loaders
        val_metric = [0] * num_of_loaders
        for i, results in enumerate(validation_step_outputs):
            for result in results:
                loss = result[0]
                cnt = result[1]
                cnt[i] += cnt
                total_metric[i] += loss * cnt
            val_metric[i] = total_metric[i] / cnt[i]
            val_metric_name = f"val_metric_{i}th_dataset"
            self.log_dict({val_metric_name: val_metric[i]}, sync_dist=True)
        self.log_dict({"val_metric": np.sum(total_metric) / np.sum(cnt)},
                      sync_dist=True)
        print("epoch loss", np.sum(total_metric) / np.sum(cnt))

    def configure_optimizers(self):
        max_iter = None

        if self.config.get("max_epochs", None):
            assert len(self.config.train_batch_sizes) == 1, "Set max_epochs only if the number of datasets is 1"
            max_iter = (self.config.max_epochs * self.config.num_training_samples_per_epoch) / (
                    self.config.train_batch_sizes[0] * torch.cuda.device_count() * self.config.get("num_nodes", 1)
            )

        if self.config.get("max_steps", None):
            max_iter = min(self.config.max_steps, max_iter) if max_iter is not None else self.config.max_steps

        assert max_iter is not None
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, betas=(0.9,0.98), eps=1e-6)
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, self.config.warmup_steps),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items["exp_name"] = f"{self.config.get('exp_name', '')}"
        items["exp_version"] = f"{self.config.get('exp_version', '')}"
        return items

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        save_path = Path(self.config.result_path) / self.config.exp_name / self.config.exp_version
        self.model.save_pretrained(save_path)
        self.model.decoder.tokenizer.save_pretrained(save_path)