from .base_dataset import BaseDataset


class VQAv2Dataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["vqav2_train", "vqav2_val"]
        elif split == "val":
            names = ["vqav2_val"]
        elif split == "test":
            names = ["vqav2_test"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )

    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]

        index, question_index = self.index_mapper[index]
        qid = self.table["question_id"][index][question_index].as_py()

        if self.split != "test":
            answers = self.table["answers"][index][question_index].as_py()
            labels = self.table["answer_labels"][index][question_index].as_py()
            scores = self.table["answer_scores"][index][question_index].as_py()
            multiple_choice_answer = self.table["multiple_choice_answer"][index].as_py()
            try:
                return_norm = self.norm_dict.words[multiple_choice_answer]["conc-m"]["sources"]["MT40k"]["scaled"] #TODO generalise this norm
            except KeyError:
                return_norm = 0.5
            ansIdx = self.answer2id[multiple_choice_answer]

            if self.loss_type in ["avsc", "avsc-scaled"]:
                assoc_tensor = self.idx2BCE_assoc_tensor[ansIdx]
                ctgrcl_tensor = self.idx2BCE_ctgrcl_tensor[ansIdx]
            else:
                assoc_tensor = []
                ctgrcl_tensor = []
        else:
            answers = list()
            labels = list()
            scores = list()
            multiple_choice_answer = list()
            return_norm = None
            assoc_tensor = list()
            ctgrcl_tensor = list()
            ansIdx = None
        return {
            "image": image_tensor,
            "text": text,
            "vqa_answer": answers,
            "vqa_labels": labels,
            "vqa_scores": scores,
            "qid": qid,
            "multiple_choice_answer": multiple_choice_answer,
            "return_norm": return_norm,
            "assoc_tensor": assoc_tensor,
            "ctgrcl_tensor": ctgrcl_tensor,
            "ansIdx": ansIdx,
        }
