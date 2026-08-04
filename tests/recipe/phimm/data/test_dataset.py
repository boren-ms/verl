from recipe.phimm.data.dataset import add_task_info


class MinimalDataset:
    def map(self, function, **kwargs):
        self.example = function({"text": "bonjour"})
        return self


def test_add_task_info_enables_language_prefix_by_default():
    dataset = add_task_info(MinimalDataset(), task="lang_asr", language="French")

    assert dataset.example["prefix"] == "Audio Language: French\n"


def test_add_task_info_allows_language_prefix_opt_out():
    dataset = add_task_info(MinimalDataset(), task="lang_asr", language="French", prefix_prob=0.0)

    assert dataset.example["prefix"] == ""