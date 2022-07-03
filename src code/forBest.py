import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataloader import RetinopathyLoader

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from datetime import datetime,timezone,timedelta
from tqdm import tqdm
import gc


def plot_Confusion_Matrix(y_true,y_pred,classes=[0,1,2,3,4],normalize=False,title=None,cmap=plt.cm.Blues,saveName='./cfm'):
    sum=np.zeros([5,5])

    for i in classes:
        for j in classes:
          sum[i,j]=np.sum((y_true==j) &(y_pred==i))
  
    if normalize:
        # normalize by truth value
        norm_val=sum.copy()
        for i in range(len(classes)):
            norm_val[:,i]/=np.sum(sum[:,i])
        sum=norm_val
        colours = cmap(norm_val)
        norm=None
    else:
        norm_val=sum-np.min(sum)
        norm_val=norm_val/np.max(norm_val)
        colours = cmap(norm_val)
        norm=Normalize(np.min(sum),np.max(sum))
    plt.clf()
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap)) 
    

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("Truth Label")
    plt.xlim([-0.5, 4.5])
    plt.ylim([-0.5, 4.5])
    for i in range(len(classes)):
        for j in range(len(classes)):
            x=[i+0.5,i-0.5,i-0.5,i+0.5]
            y=[j+0.5,j+0.5,j-0.5,j-0.5]
            plt.fill(x, y,color=colours[i,j])
            plt.text(i,j,f'{sum[i,j]:.2f}')

    plt.savefig(f'{saveName}.png')
    return


def train(model,train_loader,test_loader,epoch=10,learnRate=0.001):
    print('start training')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learnRate,momentum = 0.9,weight_decay = 0.0005)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learnRate,weight_decay = 0.0005)

    running_loss = 0.
    train_record,test_record=[],[]

    best_score=0
    best_model=None
    
    #gc.collect()
    torch.cuda.empty_cache()

    for epoch_time in range(epoch):
        running_loss=0
        model.train()
        with torch.enable_grad():
            for i,train_data in enumerate(train_loader):
                # image & label
                data,label=train_data
                # Zero gradients for every batch
                optimizer.zero_grad()

                # Make predictions for this batch
                data=data.to(device)
                label=label.to(device)
                outputs = model(data)

                # Compute the loss and its gradients
                loss = loss_fn(outputs, label)
                loss.backward()

                # Adjust learning weights
                optimizer.step()

                # Gather data and report
                running_loss += loss.item()
            
        
        torch.save(model.state_dict(), './current_model')
        
        # get accuracy
        with torch.no_grad():
            model.eval()
            acc=0        
            # train set
            for i,train_data in enumerate(train_loader):
                data,label=train_data
                data=data.to(device)
                label=label.to(device)
                outputs = model(data)
                outputs=torch.argmax(outputs, dim=1)
                acc+=torch.sum(outputs == label).item()
            train_acc=acc/28099*100
                
            # test set
            acc=0
            for i,train_data in enumerate(test_loader):
                data,label=train_data
                data=data.to(device)
                label=label.to(device)
                outputs = model(data)
                outputs=torch.argmax(outputs, dim=1)
                acc+=torch.sum(outputs == label).item()

            test_acc=acc/7025*100
            train_record.append(train_acc)
            test_record.append(test_acc)

            print(f'epoch {epoch_time+1:3} loss: {running_loss:.6f} train:{train_acc:.4f} test:{test_acc:.4f}')
            if test_acc>best_score:
                torch.save(model, './best_model')
                best_model=model
                best_score=test_acc


    print(f'max test score= {best_score}')
    return model,best_model,train_record,test_record,best_score


@torch.no_grad()
def test(model,title='confusion matrix',saveName='./cfm'):
    '''
    param:
        loadmodel: path to load model
        saveName: the file name to save confusion matrix figure
    '''
    #model=torch.load(model).to(device)
    model.eval()
    #print(f'load model from {loadmodel}')
    y_truth=np.empty([1,1])
    y_predict=np.empty([1,1])

    tsfm_test = transforms.Compose([
        transforms.ToTensor()
    ])
    test_set=RetinopathyLoader('./data',transform=tsfm_test,mode='test')
    test_loader = DataLoader(test_set,batch_size=8, shuffle=False, num_workers=2)
    acc=0
    for i,train_data in tqdm(enumerate(test_loader)):
        data,label=train_data
        data=data.to(device)
        l=label.numpy()
        label=label.to(device)
        
        outputs = model(data)
        outputs=torch.argmax(outputs, dim=1)
        acc+=torch.sum(outputs == label).item()
    
        y_truth=np.concatenate((y_truth, l), axis=None)
        pre=outputs.cpu().numpy()
        y_predict=np.concatenate((y_predict, pre), axis=None)
        
    plot_Confusion_Matrix(y_truth,y_predict,title=title,saveName=f'{saveName}_norm',normalize=True)
    plot_Confusion_Matrix(y_truth,y_predict,title=title,saveName=saveName,normalize=False)
    test_acc=acc/7025*100
    print(f'test accuracy: {test_acc}')
    return



def main():    
    # create save directory
    local=timezone(timedelta(hours=8))
    now = datetime.now().astimezone(local)
    current_time = now.strftime("%d_%H-%M-%S")
    modelSaveDir=f'./{current_time}'
    if not os.path.exists(modelSaveDir):
        os.makedirs(modelSaveDir)
    
    # hyper parameters
    lr=0.001
    batch_size=8
    trainEpoch_for18=20
    trainEpoch_for50=10
    
    # define transform
    tsfm_train = transforms.Compose([
        transforms.RandomRotation(20,expand=False),# rotate between degree
        transforms.RandomVerticalFlip(p=0.5),
        #transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    tsfm_test = transforms.Compose([
        transforms.ToTensor()
    ])

    print(tsfm_test)
    print(tsfm_train)

    f=open(f'{modelSaveDir}/tsfm.txt','w')
    f.write(str(tsfm_train))
    f.write(str(tsfm_test))
    f.close()


    # create dataset
    train_set=RetinopathyLoader('./data',transform=tsfm_train,mode='train')
    test_set=RetinopathyLoader('./data',transform=tsfm_test,mode='test')
    
    # create dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set,batch_size=batch_size, shuffle=False, num_workers=2)

    f=open('./record.txt','a')
    f.write(f'\n{modelSaveDir:15}batchsize: {batch_size:3}  lr: {lr} Epoch for 18: {trainEpoch_for18} Epoch for 50: {trainEpoch_for50}\n')
    f.close()

    '''# ResNet18 
    
    X=[(x+1) for x in range(trainEpoch_for18)]
    # using pretrained weight
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).to(device)
    model,best_model,train_record,test_record,best_score=train(model,train_loader,test_loader,epoch=trainEpoch_for18,learnRate=lr)
    name='pretrain_ResNet_18'
    torch.save(model,f'{modelSaveDir}/{name}')
    torch.save(best_model,f'{modelSaveDir}/best_{name}')
    plt.plot(X,train_record,'-',label=f"train(with pretrained)")
    plt.plot(X,test_record,'-',label=f"test(with pretrained)")
    
    # save in file
    f=open(f'{modelSaveDir}/{name}_record.txt','w')
    f.write(str(train_record))
    f.write('\n')
    f.write(str(test_record))
        
    f=open('./record.txt','a')
    f.write(f'{name:20} {test_record[-1]:.6f} best: {best_score}\n')
    f.close()

    plt.title("Result (ResNet18)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy(%)")
    plt.legend()
    plt.savefig(f'{modelSaveDir}/ResNet18')
    
    test(model,title='ResNet18_pretrain',saveName='ResNet18')
    test(best_model,title='ResNet18_pretrain',saveName='best_ResNet18')
    
    '''


    plt.clf()
    # ResNet50 
    X=[(x+1) for x in range(trainEpoch_for50)]

    # using pretrained weight
    model=torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).to(device)

    
    model,best_model,train_record,test_record,best_score=train(model,train_loader,test_loader,epoch=trainEpoch_for50,learnRate=lr)
    name='pretrain_ResNet_50'
    torch.save(model,f'{modelSaveDir}/{name}')
    torch.save(best_model,f'{modelSaveDir}/best_{name}')
    plt.plot(X,train_record,'-',label=f"train(with pretrained)")
    plt.plot(X,test_record,'-',label=f"test(with pretrained)")
    # save in file
    f=open(f'{modelSaveDir}/{name}_record.txt','w')
    f.write(str(train_record))
    f.write('\n')
    f.write(str(test_record))
    f.close()

    f=open('./record.txt','a')
    f.write(f'{name:20} {test_record[-1]:.6f} best: {best_score}\n')
    f.close()  

    plt.title("Result (ResNet50)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy(%)")
    plt.legend()
    plt.savefig(f'{modelSaveDir}/ResNet50')
    test(model,title='ResNet50_pretrain',saveName='ResNet50')
    test(best_model,title='ResNet50_pretrain',saveName='best_ResNet50')



if __name__=="__main__":
    # using GPU
    os.environ['CUDA_LAUNCH_BLOCKING']='1'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'using {device}')


    main()
    
