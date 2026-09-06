import os
import random
from PIL import Image

import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


class Rain100LDataset(Dataset):
    """
    Dataset loader for Rain100L image deraining dataset.

    input  folder : rainy images
    target folder : clean images
    """

    def __init__(self, root_dir, split="train"):

        self.input_dir = os.path.join(root_dir, "input")
        self.target_dir = os.path.join(root_dir, "target")

        self.images = sorted(
            os.listdir(self.input_dir)
        )

        if split == "train":
            self.images = self.images[:80]

        elif split == "val":
            self.images = self.images[80:]

        self.split = split


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):

        img_name = self.images[idx]

        rainy_path = os.path.join(
            self.input_dir,
            img_name
        )

        clean_path = os.path.join(
            self.target_dir,
            img_name
        )


        rainy = Image.open(rainy_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")


        # Resize both images equally
        rainy = TF.resize(rainy, (256, 256))
        clean = TF.resize(clean, (256, 256))


        # Apply augmentation only for training
        if self.split == "train":

            # Same horizontal flip
            if random.random() > 0.5:
                rainy = TF.hflip(rainy)
                clean = TF.hflip(clean)


            # Same rotation
            angle = random.uniform(-10, 10)

            rainy = TF.rotate(rainy, angle)
            clean = TF.rotate(clean, angle)


        # Convert PIL -> Tensor
        rainy = TF.to_tensor(rainy)
        clean = TF.to_tensor(clean)


        return rainy, clean
