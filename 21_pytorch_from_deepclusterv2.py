import random
import argparse
from logging import getLogger
from PIL import ImageFilter
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import numpy as np

# practice torchvision...

# torchvision.datasets - built-in datasets and classes for rolling your own
# torch.utils.datasets.ImageFolder - A generic data loader where the images are arranged 
# torch.utils.data.distributed.DistributedSampler - Sampler that restricts data loading to a subset of the dataset.
# torch.utils.data.DataLoader = iterable over a dataset - load multiple samples in parallel using workers

parser = argparse.ArgumentParser(description="Implementation of DeepCluster-v2")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")

#########################
## dcv2 specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")


#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")

#########################
#### other parameters ###
#########################
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")



class MultiCropDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
    ):
        super(MultiCropDataset, self).__init__(data_path)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        print("the number of samples is ", len(self.samples))
        self.return_index = return_index

        # color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            colordistortion = get_color_distortion()
            pilrandomgaussianblur = PILRandomGaussianBlur()
            # trans.extend([transforms.Compose([
            #     randomresizedcrop,
            #     transforms.RandomHorizontalFlip(p=0.5),
            #     transforms.Compose(color_transform),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=mean, std=std)])
            # ] * nmb_crops[i])

            comp = transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                colordistortion,
                pilrandomgaussianblur,
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])

            trans.extend([comp])

            # trans.extend([transforms.Compose([
            #     randomresizedcrop,
            #     transforms.RandomHorizontalFlip(p=0.5),
            #     transforms.Compose(color_transform),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=mean, std=std)])
            # ] * nmb_crops[i])

        self.trans = trans

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

import torchvision.transforms.functional as F
plt.rcParams["savefig.bbox"] = 'tight'
def myshowgrid(imgs):
    # if not isinstance(imgs, list):
    #     imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        # img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def myshowgrid2(imgs):
    # https://matplotlib.org/stable/gallery/axes_grid1/simple_axesgrid.html
    fig = plt.figure(figsize=(8.0, 8.0))

    nimages = len(imgs)
    ncols = 2
    nrows = ((nimages-1)//ncols) + 1
    # grid = ImageGrid(
    #     fig,
    #     111,
    #     nrows_ncols=(nrows,ncols),
    #     axes_pad=0.1,
    #     label_mode="L")
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(nrows,ncols))


    # vmin = 200
    # vmax = 200
    # for ax, im in zip(grid, imgs):
    #     ax.imshow(im, origin='lower', vmin=vmin, vmax=vmax)

    for ax, im in zip(grid, imgs):
        ax.imshow(im)

    plt.show()


def plot3(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()



def main():

    global args
    args = parser.parse_args()

    logger = getLogger()



    args.data_path = 'D:\\dataset_imagenet\\QUICK_SUBSET'
    args.nmb_crops=[2, 4]
    args.size_crops=[160, 96]
    args.min_scale_crops=[0.08, 0.05]
    args.max_scale_crops=[1., 0.14]
    args.crops_for_assign=[0, 1]

    # build data
    train_dataset = MultiCropDataset(
        args.data_path,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
        return_index=True,
    )
    # sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    sampler = torch.utils.data.SequentialSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))


    # labels_map = {
    #     0: "T-Shirt",
    #     1: "Trouser",
    #     2: "Pullover",
    #     3: "Dress",
    #     4: "Coat",
    #     5: "Sandal",
    #     6: "Shirt",
    #     7: "Sneaker",
    #     8: "Bag",
    #     9: "Ankle Boot",
    # }
    # figure = plt.figure(figsize=(8, 8))
    # cols, rows = 3, 3
    # for i in range(1, cols * rows + 1):
    #     sample_idx = torch.randint(len(training_data), size=(1,)).item()
    #     img, label = training_data[sample_idx]
    #     figure.add_subplot(rows, cols, i)
    #     plt.title(labels_map[label])
    #     plt.axis("off")
    #     plt.imshow(img.squeeze(), cmap="gray")
    # plt.show()

    # print(type(train_dataset))
    # for i in range(1, cols * rows + 1):
    #     sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    #     img, label = train_dataset[sample_idx]
    #     print(type(img))
    #     print(type(label))
    #     print(len(label))
    #     figure.add_subplot(rows, cols, i)
    #     # plt.title(labels_map[label])
    #     plt.axis("off")
    #     plt.imshow(img, cmap="gray")
    # plt.show()

    im_min = 11
    im_max = 12
    sample = train_dataset[im_min:im_max]
    print(type(sample))
    image_number = sample[0]
    image_datas = sample[1]
    print("image number ", image_number)
    print("image datas ", type(image_datas))
    for x, image_data in enumerate(image_datas):
        print("image ", x, " type ", type(image_data), " shape ", image_data.shape)

    img0 = sample[1][0]
    img0b = torch.movedim(img0, (0,1,2), (2, 0, 1))
    img1 = sample[1][1]
    img1b = torch.movedim(img1, (0,1,2), (2, 0, 1))

    img_list = [img0b,img1b]
    # # grid = make_grid(img_list)
    # # myshowgrid(grid)
    # myshowgrid(img_list)

    # plot3(img_list)
    # print("here")
    # # plt.imshow(img2)
    # # plt.show()

    myshowgrid2(img_list)


    # print(type(train_dataset[im][0]))
    # print("imager number ", train_dataset[im][0])
    # print(type(train_dataset[im][1]))
    # print(len(train_dataset[im][1]))
    # for z in range(0,6):
    #     print("----", z)
    #     print(type(train_dataset[im][1][z]))
    # print(train_dataset[im][1][z].size())

if __name__ == "__main__":
    main()
    print("done")
