import yaml
from PIL import Image
from data.base_dataset import BaseDataset

class ACDC(BaseDataset):
    def __init__(self, path_list, config_file = "./dataset/CityScapes/cityscape_copy.yaml", transform = None,
                data_set = 'val', seed=None, img_size=768, interpolation=Image.BILINEAR, color_pallete = 'city'):
        super().__init__(path_list, transform, data_set, seed, img_size, interpolation, color_pallete)
        with open(config_file, 'r') as stream:
            cityyaml = yaml.safe_load(stream)
        self.learning_map = cityyaml['learning_map']

        self.masks = [
            path.replace("/rgb_anon/", "/gt/")
            .replace("_rgb_", "_gt_")
            .replace("_anon.", "_labelIds.")
            for path in self.imgs
        ]