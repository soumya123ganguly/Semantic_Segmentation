from torch import optim

from basic_fcn import *
from resnet34_fcn import *
import time
from torch.utils.data import DataLoader
import torch
import gc
import voc
import torchvision.transforms as standard_transforms
import util
import numpy as np
import matplotlib.pyplot as plt
from focal_loss import FocalLoss

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.RandomHorizontalFlip(),
        standard_transforms.RandomVerticalFlip(),
        standard_transforms.RandomAffine((5, 10)),
        standard_transforms.RandomCrop((220, 220), padding=2),
        standard_transforms.Normalize(*mean_std)
    ])

target_transform = MaskToTensor()

train_dataset =voc.VOC('train', transform=input_transform, target_transform=target_transform)
val_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform)
test_dataset = voc.VOC('test', transform=input_transform, target_transform=target_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size= 16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= 16, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size= 16, shuffle=False)

#TODO Get class weights
def getClassWeights(n_class=21):
  w = torch.ones((n_class,))
  for (_, labels) in train_loader:
    for i in range(n_class):
      w[i] = torch.where(labels==i,1,0).sum(dtype=torch.float)
  return (w.pow_(-1))/(w.pow_(-1)).sum()

class_weights = getClassWeights()

early_stop = False

epochs = 100

exp_name = "cosine_anhealing_leaky_relu_lr_l2_0.005"

n_class = 21
patience = 10
epochs_wait = 0
best_valid_loss = 10000
fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights)
train_epoch_loss = []
valid_epoch_loss = []

#device =  "gpu" # TODO determine which device to use (cuda or cpu)
optimizer = optim.Adam(fcn_model.parameters(), lr=0.0005, weight_decay=0.0005) # TODO choose an optimizer
#criterion = FocalLoss() # TODO Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
criterion = torch.nn.CrossEntropyLoss() # TODO Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fcn_model.to(device)
# TODO transfer the model to the device

# TODO
def train():
    best_iou_score = 0.0
    best_acc_score = 0.0

    for epoch in range(epochs):
      if early_stop:
        break
      ts = time.time()
      train_loss = 0
      num_iter = 0
      for iter, (inputs, labels) in enumerate(train_loader):
        # TODO  reset optimizer gradients
        
        # both inputs and labels have to reside in the same device as the model's
        inputs =  inputs.to(device)# TODO transfer the input to the same device as the model's
        labels = labels.to(device)  # TODO transfer the labels to the same device as the model's

        outputs = fcn_model(inputs) # TODO  Compute outputs. we will not need to transfer the output, it will be automatically in the same device as the model's!

        loss = criterion(outputs, labels)  #TODO  calculate loss

        # TODO  backpropagate

        # Compute the gradients
        loss.backward()


        # TODO  update the weights

        # Update the model parameters
        optimizer.step()

        # Clear the gradients
        optimizer.zero_grad()
        scheduler.step()
        train_loss += loss.item()
        num_iter += 1
        if iter % 10 == 0:
          print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
      train_epoch_loss.append(train_loss/num_iter)
      print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
      current_miou_score, current_pacc_score = val(epoch)

      if current_miou_score > best_iou_score:
        best_iou_score = current_miou_score
        best_acc_score = current_pacc_score
        # save the best model
        torch.save(fcn_model.state_dict(), "model/{}.pth".format(exp_name))
    print("best_valid_metrics: ", best_acc_score, best_iou_score)

 #TODO
def val(epoch):
    global best_valid_loss, epochs_wait, early_stop
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing
      valid_loss = 0
      pacc = 0
      miou = 0
      num_iter = 0
      for iter, (inputs, labels) in enumerate(val_loader):
        inputs =  inputs.to(device)
        labels = labels.to(device)
        outputs = fcn_model(inputs)
        loss = criterion(outputs, labels)
        valid_loss += loss.item()
        pacc += util.pixel_acc(outputs, labels).item()
        miou += util.iou(outputs, labels).item()
        num_iter += 1
      valid_epoch_loss.append(valid_loss/num_iter)
      print(f"Loss at epoch: {epoch} is {valid_loss/num_iter}")
      print(f"IoU at epoch: {epoch} is {miou/num_iter}")
      print(f"Pixel acc at epoch: {epoch} is {pacc/num_iter}")

    fcn_model.train() #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!
    if valid_loss/num_iter > best_valid_loss:
      epochs_wait += 1
    else:
      best_valid_loss = valid_loss/num_iter
      epochs_wait = 0
    if epochs_wait == patience:
      early_stop = True
    return miou/num_iter, pacc/num_iter

 #TODO
def modelTest():
    fcn_model.load_state_dict(torch.load("model/{}.pth".format(exp_name)))
    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !
    
    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing
      pacc = 0
      miou = 0
      num_iter = 0
      for iter, (inputs, labels) in enumerate(test_loader):
        inputs =  inputs.to(device)
        labels = labels.to(device)
        outputs = fcn_model(inputs)
        pacc += util.pixel_acc(outputs, labels).item()
        miou += util.iou(outputs, labels).item()
        num_iter += 1
        idx_iter = 3
        idx = 9
        if iter == idx_iter:
          imgct = labels[idx].cpu().numpy()
          imgcy = np.argmax(outputs[idx].cpu().numpy(), axis=0)
          imgt = np.zeros((imgct.shape[0], imgct.shape[1], 3))
          imgy = np.zeros((imgcy.shape[0], imgcy.shape[1], 3))
          for i in range(imgct.shape[0]):
            for j in range(imgct.shape[1]):
              imgt[i, j] = voc.palette[3*imgct[i, j]:3*imgct[i, j]+3]
              imgy[i, j] = voc.palette[3*imgcy[i, j]:3*imgcy[i, j]+3]
          imgt = imgt.astype("uint8")
          imgy = imgy.astype("uint8")
          plt.imsave("imgs/grt_{}.png".format(exp_name), imgt)         
          plt.imsave("imgs/gry_{}.png".format(exp_name), imgy)         
      print("test_metrics: ", 
            pacc/num_iter,
            miou/num_iter)

    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train()
    modelTest()
    plt.figure()
    plt.plot(np.arange(len(train_epoch_loss)), train_epoch_loss, label="train")
    plt.plot(np.arange(len(valid_epoch_loss)), valid_epoch_loss, label="validation")
    plt.title("Base Line")
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.savefig("plots/{}.png".format(exp_name))

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()
