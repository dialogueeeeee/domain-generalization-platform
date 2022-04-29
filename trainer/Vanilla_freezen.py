from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import open_specified_layers
# from tools.freezen_specified_layers import freezen_specified_layers 


@TRAINER_REGISTRY.register()
class Vanilla_freezen(TrainerX):
    """Vanilla baseline : can freezen certain layers."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.open_layers = ['classifier']

    def forward_backward(self, batch):
        open_specified_layers(self.model, self.open_layers)
        # freezen_specified_layers(self.model, self.freezen_layers)
        input, label = self.parse_batch_train(batch)
        output = self.model(input)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
