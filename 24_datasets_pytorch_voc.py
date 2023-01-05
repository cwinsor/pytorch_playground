import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data

import matplotlib.pyplot as plt
import numpy as np

# collate...
# https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3?u=ptrblck
def my_collate(batch):
    print("here")
    data = [item[0] for item in batch]
    # target = [item[1] for item in batch]
    target = [ [123,456] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

def main():
    # transforms API is at https://pytorch.org/vision/stable/transforms.html
    # Compose allows chaining together multiple transforms
        
    batch_size = 4
    num_workers = 1
    # datafolder = 'C:\\Users\\chris\\Downloads\\ILSVRC\\Data\\CLS-LOC\\train'
    # datafolder = 'C:\\Users\\chris\\Downloads\\dataset_VOC_07\\VOCdevkit\\VOC2007\\JPEGImages'
    # datafolder = 'C:\\Users\\chris\\Downloads\\dataset_VOC_07'
    datafolder = 'C:\\Users\\chris\\Downloads\\dataset_VOC07b'
    # datafolder = 'C:\\Users\\chris\\Downloads\\dataset_FashionMNIST'
    # datafolder = 'C:\\Users\\chris\\Downloads\\Flowers102'

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # train_dataset = datasets.ImageFolder(
    #     root=datafolder,
    #     transform=train_transform)

    # train_dataset = torchvision.datasets.Flowers102(
    #     root=datafolder,
    #     split='train',
    #     transform=train_transform,
    #     download=True)

    train_dataset = datasets.VOCDetection(
        root=datafolder,
        year = '2007',
        image_set = 'train',
        download = True,
        transform = train_transform)
        # year = '2012',

    # sampler = data.SequentialSampler(
    #         data_source=train_dataset)

    # def collate_fn(batch):
    #     return tuple(zip(*batch))

    # def collate_fn(batch):
    #     return tuple(zip(*batch))



    trainloader = data.DataLoader(
        dataset=train_dataset,
        shuffle=False,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=my_collate)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # print(trainset)
    # print(testset)

    # function to show an image
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # # get some random training images
    # for (img, anno) in trainloader:
    #     x = torch.stack(img)
    #     out = model(x)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    print(len(images))
    print(labels)
  
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


if __name__ == "__main__":
        main()