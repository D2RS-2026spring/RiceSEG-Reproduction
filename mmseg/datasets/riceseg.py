# mmseg/datasets/riceseg.py
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class RiceSegDataset(BaseSegDataset):
    """RiceSeg dataset.
    
    The dataset includes 6 categories: Background, Green vegetation, 
    Senescent vegetation, Panicle, Weeds, Duckweed.
    """
    
    # 定义类别和对应的可视化颜色 (RGB 格式)
    METAINFO = dict(
        classes=('Background', 'Green vegetation', 'Senescent vegetation', 
                 'Panicle', 'Weeds', 'Duckweed'),
        palette=[
            [0, 0, 0],       # Background (黑色)
            [0, 255, 0],     # Green vegetation (绿色)
            [139, 69, 19],   # Senescent vegetation (棕色/衰老植被)
            [255, 215, 0],   # Panicle (金色/稻穗)
            [255, 0, 0],     # Weeds (红色/杂草)
            [0, 255, 255]    # Duckweed (青色/浮萍)
        ]
    )

    def __init__(self,
                 img_suffix='.jpg',      # 你的原图后缀，如果是 .png 请修改这里
                 seg_map_suffix='.png',  # 你的标签(Mask)后缀，通常是 .png
                 reduce_zero_label=False, # 如果你的 Background 标签值是 0，这里保持 False
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)