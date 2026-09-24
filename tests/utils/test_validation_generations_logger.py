from verl.utils.tracking import ValidationGenerationsLogger


class _FakeTable:
    def __init__(self, columns, data=None):
        self.columns = columns
        self.data = list(data or [])

    def add_data(self, *row):
        self.data.append(list(row))


class _FakeWandb:
    Table = _FakeTable

    def __init__(self):
        self.logged = []

    def log(self, data, step):
        self.logged.append((data, step))


def test_wandb_validation_generations_include_ground_truth():
    fake_wandb = _FakeWandb()
    logger = ValidationGenerationsLogger()

    logger._log_generations_to_wandb(
        samples=[("prompt", "prediction", "reference", 0.75)],
        step=3,
        wandb=fake_wandb,
    )

    table = fake_wandb.logged[0][0]["val/generations"]
    assert table.columns == ["step", "input_1", "output_1", "gt_1", "score_1"]
    assert table.data == [[3, "prompt", "prediction", "reference", 0.75]]
