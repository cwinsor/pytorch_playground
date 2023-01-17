
from CW:

The code is a copy of https://github.com/pytorch/vision/blob/main/references/detection
It is infrastructure to train and run transfer learning in vision.
This PyTorch code supports selected models and datasets (not all but
many good examples).  It is functioning PyTorch code for training/baseline.

For pre-testing the COCO dataset and installing "pycocotools":
    run the "cocoapi" using jupyter notebook (on windows).  See:
    https://github.com/cocodataset/cocoapi
    https://github.com/cocodataset/cocoapi/issues/272

REFERENCE:
https://github.com/pytorch/vision/blob/main/references/detection/README.md

MODELS AND PRE-TRAINED WEIGHTS
https://pytorch.org/vision/main/models.html#models-and-pre-trained-weights

TRAINING REFERENCES
https://pytorch.org/vision/main/training_references.html
with vision "detection" referenced as:
https://github.com/pytorch/vision/tree/main/references/detection

To get it working:
a) Use WSL. torchrun is only linux, or it seems that way.
b) need to set environment variable  export PYTHONPATH='/mnt/d/code_pytorch/' so it can see the /vision utils
c) edit my_train.py to set data_path parser.add_argument("--data-path", default="/mnt/d/dataset_coco_2017/coco/", type=str, help="dataset path")


Example command line:

coco:
torchrun --nproc_per_node=1 my_train.py --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26  --lr-steps 16 22 --aspect-ratio-group-factor 3 --weights-backbone ResNet50_Weights.IMAGENET1K_V1
failure signature:
  File "/mnt/d/code_pytorch/my_playground/62_pytorch_vision_ref_detection/detection/coco_utils.py", line 224, in __getitem__
    img, target = self._transforms(img, target)
TypeError: __call__() takes 2 positional arguments but 3 were given






torchrun --nproc_per_node=1 my_train.py --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26  --lr-steps 16 22 --aspect-ratio-group-factor 3 --weights-backbone ResNet50_Weights.IMAGENET1K_V1

torchrun --data-path=/mnt/d/dataset_coco_2017/coco --nproc_per_node=1 my_train.py --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26  --lr-steps 16 22 --aspect-ratio-group-factor 3 --weights-backbone ResNet50_Weights.IMAGENET1K_V1


--data-path /mnt/d/dataset_coco_2017/coco

`--data-path=/path/to/coco/dataset`


Some helpful debug references:

TORCH DISTRIBUTED ELASTIC
https://pytorch.org/docs/stable/distributed.elastic.html

TORCH DISTRIBUTED ELASTIC QUICKSTART
https://pytorch.org/docs/stable/elastic/quickstart.html

Torchrun (elastic launch)
https://pytorch.org/docs/stable/elastic/run.html