import os
import pickle
import yaml

from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class OrchardFruit(DatasetBase):

    dataset_dir = "orchardfruit"
    new_classes = ["abiu", "gac", "great_morinda", "grumichama", "maprang", "monk_fruit", "pond_apple", "wax_apple", "yangmei"]

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "features.yaml")
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.read_data(classnames, "val")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

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
        train, test = self.subsample_classes(train, test, subsample=subsample)

        super().__init__(train_x=train, val=test, test=test)

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
                classnames[folder] = ", ".join([classes[folder][f] if f != "name" else folder for f in feature])
        return classnames

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args

        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        import math

        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:39]  # take the first half
        else:
            selected = labels[39:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}

        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(impath=item.impath, label=relabeler[item.label], classname=item.classname)
                dataset_new.append(item_new)
            output.append(dataset_new)

        return output

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = []
        items = []
        for classes in ["base", "new"]:
            path = os.path.join(split_dir, classes)
            f = sorted(f.name for f in os.scandir(path) if f.is_dir())
            # folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
            for label, folder in enumerate(f, len(folders)):
                imnames = listdir_nohidden(os.path.join(path, folder))
                classname = classnames[folder]
                for imname in imnames:
                    impath = os.path.join(path, folder, imname)
                    item = Datum(impath=impath, label=label, classname=classname)
                    items.append(item)
            folders += f
        return items
