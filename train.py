from torch import optim

from basic_fcn import *
import time
from torch.utils.data import DataLoader
import torch
import gc
import voc
import torchvision.transforms as standard_transforms
import util
import numpy as np
import matplotlib.pyplot as plt

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases

#TODO Get class weights
def getClassWeights():
    raise NotImplementedError


mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

target_transform = MaskToTensor()

train_dataset =voc.VOC('train', transform=input_transform, target_transform=target_transform)
val_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform)
test_dataset = voc.VOC('test', transform=input_transform, target_transform=target_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size= 16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= 16, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size= 16, shuffle=False)

epochs = 30

n_class = 21
fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights)
train_epoch_loss = []
valid_epoch_loss = []
valid_epoch_loss = []
mean_iou_scores = []
accuracy = []

#device =  "gpu" # TODO determine which device to use (cuda or cpu)

optimizer = optim.Adam(fcn_model.parameters()) # TODO choose an optimizer
criterion = torch.nn.CrossEntropyLoss() # TODO Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fcn_model.to(device)
# TODO transfer the model to the device


# TODO
def train():
    best_iou_score = 0.0
    best_acc_score = 0.0

    for epoch in range(epochs):
      ts = time.time()
      total_loss = 0
      total_accuracy = 0
      total_iou = 0
      num_batches = 0
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
        total_loss += loss.item()
        total_accuracy += util.pixel_acc(outputs, labels)
        total_iou += util.iou(outputs, labels)
        num_batches += 1
        if iter % 10 == 0:
          print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
      train_epoch_loss.append(total_loss/num_batches)
      print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

      current_miou_score, current_acc_score = val(epoch)

      if current_miou_score > best_iou_score:
        best_iou_score = current_miou_score
      if current_acc_score > best_acc_score:
        best_acc_score = current_acc_score
      # save the best model
      # save the model
      torch.save(fcn_model.state_dict(), "./t.pth")
    print("best_valid_metrics: ", best_acc_score, best_iou_score)

 #TODO
def val(epoch):
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing
      total_loss = 0
      total_accuracy = 0
      total_iou = 0
      num_batches = 0
      for iter, (inputs, labels) in enumerate(val_loader):
        inputs =  inputs.to(device)
        labels = labels.to(device)
        outputs = fcn_model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        total_accuracy += util.pixel_acc(outputs, labels).item()
        total_iou += util.iou(outputs, labels).item()
        num_batches += 1
      valid_epoch_loss.append(total_loss/num_batches)
      accuracy.append(total_accuracy/num_batches)
      mean_iou_scores.append(total_iou/num_batches)
      print(f"Loss at epoch: {epoch} is {total_loss/num_batches}")
      print(f"IoU at epoch: {epoch} is {total_iou/num_batches}")
      print(f"Pixel acc at epoch: {epoch} is {total_accuracy/num_batches}")

    fcn_model.train() #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return total_iou/num_batches, total_accuracy/num_batches

 #TODO
def modelTest():
    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing
      total_loss = 0
      total_accuracy = 0
      total_iou = 0
      num_batches = 0
      for iter, (inputs, labels) in enumerate(test_loader):
        inputs =  inputs.to(device)
        labels = labels.to(device)
        outputs = fcn_model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        total_accuracy += util.pixel_acc(outputs, labels).item()
        total_iou += util.iou(outputs, labels).item()
        num_batches += 1
      print("test_metrics: ", 
            total_accuracy/num_batches,
            total_iou/num_batches)

    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

if __name__ == "__main__":

    val(0)  # show the accuracy before training
    train()
    modelTest()
    plt.figure()
    plt.plot(np.arange(len(train_epoch_loss)), train_epoch_loss)
    plt.plot(np.arange(len(valid_epoch_loss)), valid_epoch_loss)
    plt.savefig("loss.png")

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()
