# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import contextlib
from detectron2.data import DatasetCatalog, MetadataCatalog
from fvcore.common.timer import Timer
# from fvcore.common.file_io import PathManager
from iopath.common.file_io import PathManager

from detectron2.data.datasets.pascal_voc import register_pascal_voc
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from .cityscapes_foggy import load_cityscapes_instances
import io
import logging

logger = logging.getLogger(__name__)

JSON_ANNOTATIONS_DIR = ""
_SPLITS_COCO_FORMAT = {}
_SPLITS_COCO_FORMAT["coco"] = {
    "coco_2017_unlabel": (
        "coco/unlabeled2017",
        "coco/annotations/image_info_unlabeled2017.json",
    ),
    "coco_2017_for_voc20": (
        "coco",
        "coco/annotations/google/instances_unlabeledtrainval20class.json",
    ),
}


def register_coco_unlabel(root):
    for _, splits_per_dataset in _SPLITS_COCO_FORMAT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            meta = {}
            register_coco_unlabel_instances(
                key, meta, os.path.join(root, json_file), os.path.join(root, image_root)
            )


def register_coco_unlabel_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root

    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_unlabel_json(json_file, image_root, name)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def load_coco_unlabel_json(
    json_file, image_root, dataset_name=None, extra_annotation_keys=None
):
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    id_map = None
    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())

    imgs = coco_api.loadImgs(img_ids)

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs), json_file))

    dataset_dicts = []

    for img_dict in imgs:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        dataset_dicts.append(record)

    return dataset_dicts


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_coco_unlabel(_root)


# ==== Predefined splits for raw cityscapes foggy images ===========
_RAW_CITYSCAPES_SPLITS = {
    # "cityscapes_foggy_{task}_train": ("cityscape_foggy/leftImg8bit/train/", "cityscape_foggy/gtFine/train/"),
    # "cityscapes_foggy_{task}_val": ("cityscape_foggy/leftImg8bit/val/", "cityscape_foggy/gtFine/val/"),
    # "cityscapes_foggy_{task}_test": ("cityscape_foggy/leftImg8bit/test/", "cityscape_foggy/gtFine/test/"),
    "cityscapes_foggy_train": ("cityscapes_foggy/leftImg8bit/train/", "cityscapes_foggy/gtFine/train/"),
    "cityscapes_foggy_val": ("cityscapes_foggy/leftImg8bit/val/", "cityscapes_foggy/gtFine/val/"),
    "cityscapes_foggy_test": ("cityscapes_foggy/leftImg8bit/test/", "cityscapes_foggy/gtFine/test/"),
}


def register_all_cityscapes_foggy(root):
    # root = "manifold://mobile_vision_dataset/tree/yujheli/dataset"
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        # inst_key = key.format(task="instance_seg")
        inst_key = key
        # DatasetCatalog.register(
        #     inst_key,
        #     lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
        #         x, y, from_json=True, to_polygons=True
        #     ),
        # )
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=False, to_polygons=False
            ),
        )
        # MetadataCatalog.get(inst_key).set(
        #     image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        # )
        # MetadataCatalog.get(inst_key).set(
        #     image_dir=image_dir, gt_dir=gt_dir, evaluator_type="pascal_voc", **meta
        # )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="coco", **meta
        )

# ==== Predefined splits for Clipart (PASCAL VOC format) ===========
def register_all_clipart(root):
    # root = "manifold://mobile_vision_dataset/tree/yujheli/dataset"
    SPLITS = [
        ("Clipart1k_train", "clipart", "train"),
        ("Clipart1k_test", "clipart", "test"),
    ]
    for name, dirname, split in SPLITS:
        year = 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"
        # MetadataCatalog.get(name).evaluator_type = "coco"

# ==== Predefined splits for Watercolor (PASCAL VOC format) ===========
def register_all_water(root):
    # root = "manifold://mobile_vision_dataset/tree/yujheli/dataset"
    SPLITS = [
        ("Watercolor_train", "watercolor", "train"),
        ("Watercolor_test", "watercolor", "test"),
    ]
    for name, dirname, split in SPLITS:
        year = 2012
        # register_pascal_voc(name, os.path.join(root, dirname), split, year, class_names=["person", "dog","bicycle", "bird", "car", "cat"])
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc_water"
        # MetadataCatalog.get(name).thing_classes = ["person", "dog","bike", "bird", "car", "cat"]
        # MetadataCatalog.get(name).thing_classes = ["person", "dog","bicycle", "bird", "car", "cat"]
        # MetadataCatalog.get(name).evaluator_type = "coco"

register_all_cityscapes_foggy(_root)
register_all_clipart(_root)
register_all_water(_root)



# ==== Predefined splits for Fetus-Dataset (COCO format) ===========

def register_all_fetus():
    basepath = '/media/Storage2/Lvxg/One_Stage_Fetus_Object_Detection_Code_v3/Dataset_Fetus_Object_Detection/'
    SPLITS = {
    # Heart
    "fetus_4c_hos1_train": ('/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Heart/c1/train.json', '/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Heart/c1/src'),
    "fetus_4c_hos1_val": ('/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Heart/c1/val.json', '/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Heart/c1/src'),
    "fetus_4c_hos1_test": ('/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Heart/c1/test.json', '/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Heart/c1/src'),
    "fetus_4c_hos2_train": ('/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Heart/c2/train.json', '/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Heart/c2/src'),
    "fetus_4c_hos2_val": ('/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Heart/c2/val.json', '/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Heart/c2/src'),
    "fetus_4c_hos2_test": ('/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Heart/c2/test.json', '/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Heart/c2/src'),\
    "fetus_4c_hos3_train": ('/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Heart/c3/train.json', '/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Heart/c3/src'),
    "fetus_4c_hos3_val": ('/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Heart/c3/val.json', '/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Heart/c3/src'),
    "fetus_4c_hos3_test": ('/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Heart/c3/test.json', '/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Heart/c3/src'),
    
    # MMWHS
    "mmwhs_ct_train":("/media/Storage1/Lvxg/mmwhs/ct_png/train/annotations.json", '/media/Storage1/Lvxg/mmwhs/ct_png/train'),
    "mmwhs_ct_val":("/media/Storage1/Lvxg/mmwhs/ct_png/val/annotations.json", '/media/Storage1/Lvxg/mmwhs/ct_png/val'),
    "mmwhs_mr_train":("/media/Storage1/Lvxg/mmwhs/mr_png/train/annotations.json", '/media/Storage1/Lvxg/mmwhs/mr_png/train'),
    "mmwhs_mr_val":("/media/Storage1/Lvxg/mmwhs/mr_png/val/annotations.json", '/media/Storage1/Lvxg/mmwhs/mr_png/val'),
    
    # EP
    "EP_public_train": ("/media/Storage2/Lvxg/EP_dataset/EP_public_annotations/train/annotation.json", "/media/Storage2/Lvxg/EP_dataset/EP_public_img"),
    "EP_public_val": ("/media/Storage2/Lvxg/EP_dataset/EP_public_annotations/val/annotation.json", "/media/Storage2/Lvxg/EP_dataset/EP_public_img"),
    "EP_public_test": ("/media/Storage2/Lvxg/EP_dataset/EP_public_annotations/test/annotation.json", "/media/Storage2/Lvxg/EP_dataset/EP_public_img"),
    "EP_ours_train": ("/media/Storage2/Lvxg/EP_dataset/EP_fetus_annotations/train/annotation.json", "/media/Storage2/Lvxg/EP_dataset/EP_fetus"),
    "EP_ours_val": ("/media/Storage2/Lvxg/EP_dataset/EP_fetus_annotations/val/annotation.json", "/media/Storage2/Lvxg/EP_dataset/EP_fetus"),
    "EP_ours_test": ("/media/Storage2/Lvxg/EP_dataset/EP_fetus_annotations/test/annotation.json", "/media/Storage2/Lvxg/EP_dataset/EP_fetus"),
    
    # Cardiac-UDA
    "Cardiac_R_train": ("/media/Storage2/Lvxg/One_Stage_Fetus_Object_Detection_Code_v3/Cardiac_UDA_5f/Site_R/train/annotations_coco.json", "/media/Storage2/Lvxg/One_Stage_Fetus_Object_Detection_Code_v3/Cardiac_UDA_5f/Site_R/train"),
    "Cardiac_R_val": ("/media/Storage2/Lvxg/One_Stage_Fetus_Object_Detection_Code_v3/Cardiac_UDA_5f/Site_R/valid/annotations_coco.json", "/media/Storage2/Lvxg/One_Stage_Fetus_Object_Detection_Code_v3/Cardiac_UDA_5f/Site_R/valid"),
    "Cardiac_R_test": ("/media/Storage2/Lvxg/One_Stage_Fetus_Object_Detection_Code_v3/Cardiac_UDA_5f/Site_R/test/annotations_coco.json", "/media/Storage2/Lvxg/One_Stage_Fetus_Object_Detection_Code_v3/Cardiac_UDA_5f/Site_R/test"),
    "Cardiac_G_train": ("/media/Storage2/Lvxg/One_Stage_Fetus_Object_Detection_Code_v3/Cardiac_UDA_5f/Site_G/train/annotations_coco.json", "/media/Storage2/Lvxg/One_Stage_Fetus_Object_Detection_Code_v3/Cardiac_UDA_5f/Site_G/train"),
    "Cardiac_G_val": ("/media/Storage2/Lvxg/One_Stage_Fetus_Object_Detection_Code_v3/Cardiac_UDA_5f/Site_G/valid/annotations_coco.json", "/media/Storage2/Lvxg/One_Stage_Fetus_Object_Detection_Code_v3/Cardiac_UDA_5f/Site_G/valid"),
    "Cardiac_G_test": ("/media/Storage2/Lvxg/One_Stage_Fetus_Object_Detection_Code_v3/Cardiac_UDA_5f/Site_G/test/annotations_coco.json", "/media/Storage2/Lvxg/One_Stage_Fetus_Object_Detection_Code_v3/Cardiac_UDA_5f/Site_G/test"),
    
    # FUSH2-Head
    "fetus_head_c1_train": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Head/c1/train.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Head/c1/src'),
    "fetus_head_c1_val": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Head/c1/val.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Head/c1/src'),
    "fetus_head_c1_test": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Head/c1/test.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Head/c1/src'),
    "fetus_head_c2_train": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Head/c2/train.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Head/c2/src'),
    "fetus_head_c2_val": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Head/c2/val.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Head/c2/src'),
    "fetus_head_c2_test": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Head/c2/test.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Head/c2/src'),
    
    # Spine
    "fetus_spine_ge_train": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/spine/GE/train.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/spine/GE/src'),
    "fetus_spine_ge_val": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/spine/GE/val.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/spine/GE/src'),
    "fetus_spine_ge_test": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/spine/GE/test.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/spine/GE/src'),
    "fetus_spine_ph_train": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/spine/PH/train.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/spine/PH/src'),
    "fetus_spine_ph_val": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/spine/PH/val.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/spine/PH/src'),
    "fetus_spine_ph_test": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/spine/PH/test.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/spine/PH/src'),
    "fetus_spine_sa_train": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/spine/SA/train.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/spine/SA/src'),
    "fetus_spine_sa_val": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/spine/SA/val.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/spine/SA/src'),
    "fetus_spine_sa_test": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/spine/SA/test.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/spine/SA/src'),
    
    # abdomen
    "fetus_abdomen_ge_train": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/abdomen/GE/train.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/abdomen/GE/src'),
    "fetus_abdomen_ge_val": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/abdomen/GE/val.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/abdomen/GE/src'),
    "fetus_abdomen_ge_test": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/abdomen/GE/test.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/abdomen/GE/src'),
    "fetus_abdomen_ph_train": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/abdomen/PH/train.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/abdomen/PH/src'),
    "fetus_abdomen_ph_val": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/abdomen/PH/val.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/abdomen/PH/src'),
    "fetus_abdomen_ph_test": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/abdomen/PH/test.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/abdomen/PH/src'),
    "fetus_abdomen_sa_train": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/abdomen/SA/train.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/abdomen/SA/src'),
    "fetus_abdomen_sa_val": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/abdomen/SA/val.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/abdomen/SA/src'),
    "fetus_abdomen_sa_test": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/abdomen/SA/test.json", '/media/Storage2/Lvxg/ToMo-UDA/dataset/abdomen/SA/src'),

    # MICCAI & CHAOS
    "Synapse_CT_train": ("/media/Storage1/Lvxg/AMO/abdomen/CT_png/train/annotations.json", "/media/Storage1/Lvxg/AMO/abdomen/CT_png/train"),
    "Synapse_CT_test": ("/media/Storage1/Lvxg/AMO/abdomen/CT_png/test/annotations.json", "/media/Storage1/Lvxg/AMO/abdomen/CT_png/test"),
    "CHAOS_MR_train":("/media/Storage1/Lvxg/AMO/abdomen/MR_png/train/annotations.json", "/media/Storage1/Lvxg/AMO/abdomen/MR_png/train"),
    "CHAOS_MR_test":("/media/Storage1/Lvxg/AMO/abdomen/MR_png/test/annotations.json", "/media/Storage1/Lvxg/AMO/abdomen/MR_png/test"),

    # FCS-4CC
    "4cc_A_train": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/4CC/A/train.json", "/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/4CC/A/src"),
    "4cc_A_val": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/4CC/A/val.json", "/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/4CC/A/src"),
    "4cc_A_test": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/4CC/A/test.json", "/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/4CC/A/src"),
    "4cc_B_train": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/4CC/B/train.json", "/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/4CC/B/src"),
    "4cc_B_val": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/4CC/B/val.json", "/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/4CC/B/src"),
    "4cc_B_test": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/4CC/B/test.json", "/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/4CC/B/src"),

    # FCS-3VT
    "3vt_A_train": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/3VT/A/train.json", "/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/3VT/A/src"),
    "3vt_A_val": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/3VT/A/val.json", "/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/3VT/A/src"),
    "3vt_A_test": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/3VT/A/test.json", "/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/3VT/A/src"),
    "3vt_B_train": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/3VT/B/train.json", "/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/3VT/B/src"),
    "3vt_B_val": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/3VT/B/val.json", "/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/3VT/B/src"),
    "3vt_B_test": ("/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/3VT/B/test.json", "/media/Storage2/Lvxg/ToMo-UDA/dataset/FCS/3VT/B/src"),

    # CRL
    "CRL_KL_train": ("/media/Storage1/Lvxg/mmwhs/CRL/KL/train.json", "/media/Storage1/Lvxg/mmwhs/CRL/KL"),
    "CRL_KL_val": ("/media/Storage1/Lvxg/mmwhs/CRL/KL/val.json", "/media/Storage1/Lvxg/mmwhs/CRL/KL"),
    "CRL_SA_train": ("/media/Storage1/Lvxg/mmwhs/CRL/SA/train.json", "/media/Storage1/Lvxg/mmwhs/CRL/SA"),
    "CRL_SA_val": ("/media/Storage1/Lvxg/mmwhs/CRL/SA/val.json", "/media/Storage1/Lvxg/mmwhs/CRL/SA"),

    }
    for key, (json_dir, img_dir) in SPLITS.items():
        register_coco_instances(key, {}, json_dir, img_dir)

register_all_fetus()