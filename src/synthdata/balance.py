"""Meeting 1: the two classical ways to add Egyptian Mau training examples.

Both functions are yours to write; `uv run synth-train --method oversample` or
`--method augment` uses them.
"""

import random

from PIL import Image
from torchvision import transforms

AUGMENT_TRANSFORM = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=8,
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1
        ),
    ]
)


def oversample(
    minority_photos: list[str], budget: int, rng: random.Random
) -> list[str]:
    """Choose which Egyptian Mau training photos to repeat.

    Return a list of exactly `budget` names taken from `minority_photos`; names
    may repeat. Repeat the photos as evenly as possible and use `rng` for every
    random choice, so a run can be reproduced from its seed.
    """
    full_repeats, remainder = divmod(budget, len(minority_photos))
    chosen = list(minority_photos) * full_repeats
    chosen.extend(rng.sample(list(minority_photos), remainder))
    rng.shuffle(chosen)
    return chosen


def augment(image: Image.Image) -> Image.Image:
    """Return a randomly transformed copy of one Egyptian Mau training photo.

    The training script calls this every time an added copy is loaded, so each
    epoch sees a new variant. Typical choices are crops, flips, rotations, and
    colour changes from `torchvision.transforms`. Keep the cat recognisable.
    """
    return AUGMENT_TRANSFORM(image)
