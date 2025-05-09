import yaml
from PIL import Image
from data.base_dataset import BaseDataset

class SemanticKITTI(BaseDataset):
    def __init__(self, path_list, config_file = "./dataset/SemanticKitti/semantickitti.yaml.yaml", transform = None,
                data_set = 'val', seed=None, img_size=768, interpolation=Image.BILINEAR, color_pallete = 'city'):
        super().__init__(path_list, transform, data_set, seed, img_size, interpolation, color_pallete)
        with open(config_file, 'r') as stream:
            cityyaml = yaml.safe_load(stream)
        self.learning_map = cityyaml['learning_map']

        self.masks = [
            path.replace("/training/image_02", "/kitti-step/panoptic_maps/"+data_set)
            for path in self.imgs
        ]

    def convert_label(self, label, inverse=False):
        label = label [0,:,:]
        temp = label.copy()*255 
        for k, v in self.learning_map.items():
            label[temp== k] = v
        return label
