import torch
from dataset import ImageDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import torchvision
import torch.optim as optim
from model import  CNN_model
import torch.nn as nn
import numpy as np
from utils_cells import calculate_precision_recall_per_class, get_accuracies_per_class
import sys
from vit import VisionTransformer
import time
import pandas as pd

torch.cuda.empty_cache()

model = VisionTransformer(image_size=32, in_channels=4, num_classes=4, hidden_dims=[16, 16])


batch_size = 64

trainset = ImageDataset(data_path='train_data')
trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, num_workers=5)


testset =ImageDataset(data_path='validation_data')

testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=False, num_workers=5)

model = model.to('cuda:0')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

recalls = []
val_recalls = []

losses = []
val_losses = []

for epoch in range(20):
    recall = []
    precision = []

    recall_val = []
    precision_val = []

    training_loss = []
    start_time = time.time()
    elapsed_time = 0
    model.train() 
    for i, data in enumerate(trainloader):

        inputs, labels = data
        labels = torch.Tensor(labels)
        
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')

        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        training_loss.append(loss.item())


        outputs = outputs.data.cpu().numpy().argmax(axis=1)
        labels = labels.data.cpu().numpy().argmax(axis=1)

        recall_per_class, precision_per_class = calculate_precision_recall_per_class(labels, outputs)
        if (i + 1) % 1000 == 0 or i == len(trainloader) - 1:
            elapsed_time = time.time() - start_time
            batches_done = i + 1
            batches_total = len(trainloader)
            batches_remaining = batches_total - batches_done
            time_per_batch = elapsed_time / batches_done
            estimated_time_remaining = time_per_batch * batches_remaining

            # Convert times to minutes
            elapsed_time_minutes = elapsed_time / 60
            estimated_time_remaining_minutes = estimated_time_remaining / 60


         
            # Print training progress and estimated time remaining on the same line
            progress_message = f'Batch {i}/{len(trainloader)},Remaining: {estimated_time_remaining_minutes:.2f}min , loss {loss.item()}, class 1: {recall_per_class[0]}, class 2: {recall_per_class[1]}, class 3: {recall_per_class[2]}, class 4: {recall_per_class[3]}'
            sys.stdout.write("\r" + progress_message)
            sys.stdout.flush()

        
        recall.append(recall_per_class)
        precision.append(precision_per_class)
        
 
    model.eval()  
    val_loss = []
    
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            labels = torch.Tensor(labels)
            inputs = inputs.to('cuda:0')
            labels = labels.to('cuda:0')


            outputs = model(inputs)
            val_loss_crt = criterion(outputs, labels)

            val_loss.append(val_loss_crt.item())

            outputs = outputs.data.cpu().numpy().argmax(axis=1)
            labels = labels.data.cpu().numpy().argmax(axis=1)
            
            recall_per_class, precision_per_class = calculate_precision_recall_per_class(labels, outputs)
            
            recall_val.append(recall_per_class)
            precision_val.append(precision_per_class)


    recall = np.mean(np.array(recall), axis=0)
    recall_val = np.mean(np.array(recall_val), axis=0)
    
    recalls.append(recall)
    val_recalls.append(recall_val)
    
    losses.append(training_loss)
    val_losses.append(val_loss)
    
    print(f'Epoch {epoch + 1}, Training loss: {np.mean(training_loss)} Validation Loss: {np.mean(val_loss)}')
    print(f'Epoch {epoch + 1}, Training Class 1: {recall[0]}, Class 2: {recall[1]}, Class 3: {recall[2]}, Class 4: {recall[3]}')
    print(f'Epoch {epoch + 1}, Validation Class 1: {recall_val[0]}, Class 2: {recall_val[1]}, Class 3: {recall_val[2]}, Class 4: {recall_val[3]}')
   
    

print('Finished Training')
print('Finished Training')


df = pd.DataFrame()

df['loss'] = np.array(losses)
df['val_loss'] = np.array(val_losses)
df.to_csv('results_loss.csv', index=False)



df['recall'] = np.array(recalls)
df['val_recall'] = np.array(val_recalls)

df.to_csv('results.csv', index=False)

torch.save(model.state_dict(),'model_vit1.pth')r





