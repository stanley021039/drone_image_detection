import os
import cv2
import torch
import numpy as np
from torch import nn
from thop import profile
import torchvision
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import DroneDataset
from torchvision.models.detection import FasterRCNN
from torch.utils.data import DataLoader
from utils import nms, mean_average_precision
save_dir = 'savemodel1008_FastRCNN'

# Initialize Dataset
full_dataset = DroneDataset(split='train', resize=(1344, 720))
public_dataset = DroneDataset(split='public', resize=(1344, 720))
train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [int(len(full_dataset)*0.8), len(full_dataset)-int(len(full_dataset)*0.8)])

# Build DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))
train_data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
valid_data_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)
public_data_loader = DataLoader(public_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)


# Check GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Establish model pretrained with Imagenet
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=5)
model.roi_heads.box_predictor.cls_score.weight.data.fill_(0)
model.roi_heads.box_predictor.cls_score.bias.data.fill_(0.01)

# Hyperparameters
num_epochs = 60
'''
inp = torch.zeros((2, 3, 720, 1344))
flops, params = profile(model, (inp,))

Total_params = 0
Trainable_params = 0
NonTrainable_params = 0

for param in model.parameters():
    mulValue = np.prod(param.size())
    Total_params += mulValue
    if param.requires_grad:
        Trainable_params += mulValue
    else:
        NonTrainable_params += mulValue

print(f'Total params: {Total_params}')
print(f'Trainable params: {Trainable_params}')
print(f'Non-trainable params: {NonTrainable_params}')
print(f'floating point operations: {flops}')
'''
model = nn.DataParallel(model, device_ids=[0])
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.004, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

# Train function
def train(epoch, trainloader, model, optimizer):
    model.train()
    itr = 1
    avg_loss = 0
    for images, targets, img_name in tqdm(trainloader, ncols=80):
        images = torch.cat([torch.unsqueeze(image, dim=0).to(device) for image in images], dim=0)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        avg_loss += loss_value

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if itr % 50 == 0:
            print(f"Iteration #{itr} train_loss: {avg_loss / itr}")

        itr += 1

    print(f"Epoch #{epoch} train_loss: {avg_loss / itr}")
    return avg_loss / itr

# Valid function
def valid(epoch, validationloader, model):
    model.eval()
    itr = 1
    avg_mAP = 0
    bbox_color = {4: (220/255, 0, 0), 1: (0, 220/255, 0), 2: (0, 0, 220/255), 3: (0, 220/255, 220/255)}
    with torch.no_grad():
        for images, targets, img_name in tqdm(validationloader, ncols=80):

            images = torch.cat([torch.unsqueeze(image, dim=0).to(device) for image in images], dim=0)
            output = model(images)
            boxess = []
            # DP時會有bug
            for idx in range(images.shape[0]):
                boxes = output[idx]['boxes'].data.cpu().numpy()
                scores = output[idx]['scores'].data.cpu().numpy()
                labels = output[idx]['labels'].data.cpu().numpy()
                boxess.append(nms(idx, boxes, scores, iou_threshold=0.5, threshold=0.4))
                if itr % 5 == 0:
                    sample = images[idx].permute(1, 2, 0).cpu().numpy().copy()
                    fog, ax = plt.subplots(1, 1, figsize=(12, 6))
                    for i, box in enumerate(boxes):
                        x0, y0, x1, y1 = [round(x) for x in box]
                        cv2.rectangle(sample,
                                      (x0, y0),
                                      (x1, y1),
                                      bbox_color[labels[i]], 2)
                    ax.set_axis_off()
                    ax.imshow(sample)
                    plt.savefig(os.path.join(save_dir, 'valid_result', str(epoch) + '_' + img_name[idx]))
                    plt.close('all')
            avg_mAP += mean_average_precision(list(boxess), list(targets), iou_threshold=0.5, num_classes=5)

            itr += 1
    print(f"Epoch #{epoch} valid_mAP: {avg_mAP / itr}")
    return avg_mAP / itr

# Test function
def test(epoch, testloader, model):
    model.eval()
    bbox_color = {4: (220/255, 0, 0), 1: (0, 220/255, 0), 2: (0, 0, 220/255), 3: (0, 220/255, 220/255)}
    with torch.no_grad():
        for images, targets, img_name in tqdm(testloader, ncols=80):

            images = torch.cat([torch.unsqueeze(image, dim=0).to(device) for image in images], dim=0)
            output = model(images)
            boxess = []
            # DP時會有bug
            for idx in range(images.shape[0]):
                boxes = output[idx]['boxes'].data.cpu().numpy()
                scores = output[idx]['scores'].data.cpu().numpy()
                labels = output[idx]['labels'].data.cpu().numpy()
                boxess.append(nms(idx, boxes, scores, iou_threshold=0.5, threshold=0.4))
                sample = images[idx].permute(1, 2, 0).cpu().numpy().copy()
                fog, ax = plt.subplots(1, 1, figsize=(12, 6))
                for i, box in enumerate(boxes):
                    x0, y0, x1, y1 = [round(x) for x in box]
                    cv2.rectangle(sample,
                                  (x0, y0),
                                  (x1, y1),
                                  bbox_color[labels[i]], 2)
                ax.set_axis_off()
                ax.imshow(sample)
                plt.savefig(os.path.join(save_dir, 'test_result', str(epoch) + '_' + img_name[idx]))
                plt.close('all')


# Start training
train_loss = []
valid_mAP = []

is_train = 0
is_valid = 0
is_test = 1
if is_train:
    for epoch in range(num_epochs):

        tra_loss = train(epoch, train_data_loader, model, optimizer)
        val_loss = valid(epoch, valid_data_loader, model)
        train_loss.append(tra_loss)
        valid_mAP.append(val_loss)

        lr_scheduler.step()
        print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])

    plt.plot(train_loss)
    plt.show()
    plt.plot(valid_mAP)
    plt.show()

    # Savemodel
    torch.save(model.state_dict(), save_dir + '/model.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'ckpt.pth')

if is_valid:
    model.load_state_dict(torch.load(save_dir + '/model.pth'))
    val_loss = valid(60, valid_data_loader, model)

if is_test:
    model.load_state_dict(torch.load(save_dir + '/model.pth'), strict=False)
    test(60, public_data_loader, model)
