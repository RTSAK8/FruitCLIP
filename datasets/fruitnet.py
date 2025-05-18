import os
import pickle
from collections import OrderedDict
import yaml
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing
from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class FruitNet(DatasetBase):
    def __init__(self, cfg):
        self.dataset_dir = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        text_file = os.path.join(self.dataset_dir, "features.yaml")
        classnames = self.read_classnames(text_file)

        train_classes = os.listdir(os.path.join(self.image_dir, "train"))
        val_classes = os.listdir(os.path.join(self.image_dir, "val"))

        train_classnames = {k: v for k, v in classnames.items() if k in train_classes}
        val_classnames = {k: v for k, v in classnames.items() if k in val_classes}

        train = self.read_data(train_classnames, "train")
        # Follow standard practice to perform evaluation on the val set
        # Also used as the val set (so evaluate the last-step model)
        test = self.read_data(val_classnames, "val")

        preprocessed = {"train": train, "test": test}

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)

        super().__init__(train_x=train, val=test, test=test)
        self.train_lab2cname, self._train_classnames = self.get_lab2cname(train)
        self.test_lab2cname, self._test_classnames = self.get_lab2cname(test)

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            classes: dict = yaml.safe_load(f)
            feature: list = classes["__base__"]
            for folder in classes.keys():
                if folder == "__base__":
                    continue
                classnames[folder] = ", ".join([classes[folder][f] for f in feature])
        return classnames

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items

    @property
    def train_classnames(self):
        return self._train_classnames

    @property
    def test_classnames(self):
        return self._test_classnames
