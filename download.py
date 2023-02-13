import torchvision

train_dataset =torchvision.datasets.VOCSegmentation(root='./data',year='2007',download=True,image_set='train')
val_dataset = torchvision.datasets.VOCSegmentation(root='./data',year='2007',download=True,image_set='val')
test_dataset = torchvision.datasets.VOCSegmentation(root='./data',year='2007',download=True,image_set='test')