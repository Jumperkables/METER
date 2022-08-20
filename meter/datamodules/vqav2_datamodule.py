from ..datasets import VQAv2Dataset
from .datamodule_base import BaseDataModule
from collections import defaultdict

import os
import sys
import json


class VQAv2DataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normonly_flag = str(args[0]["normonly_flag"])
        self.norm_clipping = args[0]["norm_clipping"]
        self.loss_type = args[0]["loss_type"]

    @property
    def dataset_cls(self):
        return VQAv2Dataset

    @property
    def dataset_name(self):
        return "vqa"

    def setup(self, stage):
        super().setup(stage)
        train_answers = self.train_dataset.table["answers"].to_pandas().tolist()
        val_answers = self.val_dataset.table["answers"].to_pandas().tolist()
        train_labels = self.train_dataset.table["answer_labels"].to_pandas().tolist()
        val_labels = self.val_dataset.table["answer_labels"].to_pandas().tolist()

        all_answers = [c for c in train_answers + val_answers if c is not None]
        all_answers = [l for lll in all_answers for ll in lll for l in ll]
        all_labels = [c for c in train_labels + val_labels if c is not None]
        all_labels = [l for lll in all_labels for ll in lll for l in ll]

        if self.data_dir == "data/vqacp2_full_arrow":
            print("THIS APPENDING IS DONE BECAUSE THESE TWO ANSWERS ARE MISSED FOR SOME REASON")
            all_answers.append('5 star')
            all_answers.append("1 world")

        self.answer2id = {ans:idx for idx, ans in enumerate(list(set(all_answers)))}
        #self.answer2id = {k: v for k, v in zip(all_answers, all_labels)}
        sorted_a2i = sorted(self.answer2id.items(), key=lambda x: x[1])
        self.num_class = max(self.answer2id.values()) + 1
        avsc_path = os.path.abspath(f"{os.path.dirname(__file__)}/../../../a_vs_c")
        sys.path.append(avsc_path)
        from datasets import set_avsc_loss_tensor
        # One line class definition to create dummy args
        dummy_args = lambda: None; dummy_args.norm_ans_only = self.normonly_flag; dummy_args.norm_clipping = self.norm_clipping
        idx2BCE_assoc_tensor, idx2BCE_ctgrcl_tensor, norm_dict = set_avsc_loss_tensor(dummy_args, self.answer2id)

        self.train_dataset.norm_dict = norm_dict
        self.val_dataset.norm_dict = norm_dict

        self.train_dataset.answer2id = self.answer2id
        self.val_dataset.answer2id = self.answer2id

        self.train_dataset.normonly_flag = self.normonly_flag
        self.val_dataset.normonly_flag = self.normonly_flag

        self.train_dataset.loss_type = self.loss_type
        self.val_dataset.loss_type = self.loss_type

        if self.loss_type in ["avsc", "avsc-scaled"]:
            # set the a_vs_c loss tensor
            self.train_dataset.idx2BCE_assoc_tensor = idx2BCE_assoc_tensor
            self.val_dataset.idx2BCE_assoc_tensor = idx2BCE_assoc_tensor
            self.train_dataset.idx2BCE_ctgrcl_tensor = idx2BCE_ctgrcl_tensor
            self.val_dataset.idx2BCE_ctgrcl_tensor = idx2BCE_ctgrcl_tensor
        
        self.id2answer = defaultdict(lambda: "unknown")
        for k, v in sorted_a2i:
            self.id2answer[v] = k


