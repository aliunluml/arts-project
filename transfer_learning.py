import torch as t
import torchvision as tv
import pandas as pd
import os
from multiprocessing import cpu_count


RANDOM_SEED=0
BATCH_SIZE=16
EPOCHS=5
LEARNING_RATE=1e-3
TRAINING_DATASET_DIRECTORY='Training'
TESTING_DATASET_DIRECTORY='Validation'
DETECTOR_DIRECTORY='pretrained'

def train(net,optimizer,objective,dataloader,device,iteration):

    net.train()
    net.to(device)

    losses=[]
    accs=[]
    for batch,labels in dataloader:
        batch=batch.to(device)
        labels=labels.to(device)

        logits = net(batch)
        nll = objective(logits,labels)

        nll.backward()
        optimizer.step()

        optimizer.zero_grad()
        loss = nll.detach().cpu()
        losses.append(loss)

        gender = t.argmax(logits,dim=1)

        acc = len(gender[gender==labels])/len(batch)
        accs.append(acc)

        iteration += 1

    epoch_loss = sum(losses)/len(losses)
    epoch_acc = sum(accs)/len(accs)

    return (epoch_loss, epoch_acc,iteration)

def test(net,objective,dataloader,device):
    net.eval()

    losses=[]
    accs=[]
    with t.no_grad():
        for batch,labels in dataloader:
            batch=batch.to(device)
            labels=labels.to(device)

            logits = net(batch)
            nll = objective(logits,labels)
            losses.append(nll.cpu())

            gender = t.argmax(logits,dim=1)

            acc = len(gender[gender==labels])/len(batch)
            accs.append(acc)

    epoch_loss = sum(losses)/len(losses)
    epoch_acc = sum(accs)/len(accs)

    return (epoch_loss, epoch_acc)


def main():
    t.backends.cudnn.benchmark=True

    t.manual_seed(RANDOM_SEED)

    if t.cuda.is_available():
        device = t.device("cuda")
    else:
        device = t.device("cpu")

    # The pretrained FSA-Net input size is 64x64
    transforms_train = tv.transforms.Compose([tv.transforms.Resize((64, 64)),tv.transforms.RandomHorizontalFlip(), tv.transforms.ToTensor(),tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transforms_test = tv.transforms.Compose([tv.transforms.Resize((64, 64)),tv.transforms.ToTensor(),tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    project_dir = os.getcwd()
    # ImageFolder object implicitly labels images with the index of their corresponding folders in alphabetical order (female=0, male=1). Please see the find_classes method within ImageFolder.
    training_dataset = tv.datasets.ImageFolder(os.path.join(project_dir,TRAINING_DATASET_DIRECTORY), transforms_train)
    testing_dataset = tv.datasets.ImageFolder(os.path.join(project_dir, TESTING_DATASET_DIRECTORY), transforms_test)

    training_dataloader = t.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=cpu_count(),pin_memory=True)
    testing_dataloader = t.utils.data.DataLoader(testing_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=cpu_count(),pin_memory=True)

    # Load a pretrained Resnet18 model and change its last fully connected layer such that it has 2 logits
    resnet = tv.models.resnet18(weights='IMAGENET1K_V1')
    num_features = resnet.fc.in_features
    resnet.fc = t.nn.Linear(num_features, 2)

    optimizer=t.optim.SGD(resnet.parameters(),lr=LEARNING_RATE,momentum=0.9)
    creloss=t.nn.CrossEntropyLoss()

    info = []
    i = 0

    for e in range(0,EPOCHS):
        train_loss,train_acc,i = train(resnet,optimizer,creloss,training_dataloader,device,i)
        test_loss,test_acc = test(resnet,creloss,testing_dataloader,device)
        epoch_info={'epoch':e,'train_loss':train_loss,'train_accuracy':train_acc,'test_loss':test_loss,'test_accuracy':test_acc}
        info.append(epoch_info)

    filename = 'resnet18-iter-'+str(i)
    df = pd.DataFrame(info)
    df.to_csv((filename+'.csv'),index=False)
    t.save(resnet.state_dict(), os.path.join(project_dir,'pretrained',filename+'.pth'))

    # Input to the model
    x = next(iter(testing_dataloader))

    # Export the model
    t.onnx.export(resnet,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  os.path.join(project_dir,'pretrained',filename+'.onnx'),   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})


if __name__ == '__main__':
    main()
