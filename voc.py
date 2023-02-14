import os
from PIL import Image
from torch.utils import data

num_classes = 21
ignore_label = 255
root = './data'

'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''


#Feel free to convert this palette to a map
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]  #3 values- R,G,B for every class. First 3 values for class 0, next 3 for
#class 1 and so on......


def make_dataset(mode):
    assert mode in ['train', 'val', 'test']
    items = []
    img_path = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
    mask_path = os.path.join(root, 'VOCdevkit', 'VOC2007', 'SegmentationClass')
    data_list = [l.strip('\n') for l in open(os.path.join(
    root, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Segmentation', '{}.txt'.format(mode))).readlines()]
    for it in data_list:
        item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
        items.append(item)
    return items


class VOC(data.Dataset):
    def __init__(self, mode, transform=None, target_transform=None):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.width = 224
        self.height = 224

    def __getitem__(self, index):

        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB').resize((self.width, self.height))
        mask = Image.open(mask_path).resize((self.width, self.height))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        print(img)
        mask[mask==ignore_label]=0

        return img, mask

    def __len__(self):
        return len(self.imgs)