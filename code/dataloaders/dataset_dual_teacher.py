"""Training samples for a 2D student and a frozen 2.5D teacher."""

from dataloaders.dataset_25d import PROMISE12DataSets25D


class PROMISE12DualTeacherDataSets(PROMISE12DataSets25D):
    """Return the centre slice and its aligned three-slice context.

    ``PROMISE12DataSets25D`` applies one shared geometric transform to all
    three slices and the centre label.  The student receives only channel 1;
    the auxiliary teacher receives the full context tensor.
    """

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        context_image = sample["image"]
        sample["image"] = context_image[1:2]
        sample["context_image"] = context_image
        return sample
