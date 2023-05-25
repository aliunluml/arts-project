import functools
import random
import torch as t
import numpy as np
import pandas as pd
import torchvision as tv
import os
import onnx
import onnxruntime
from dataset import PainterByNumbers
from multiprocessing import cpu_count


RANDOM_SEED=0
BATCH_SIZE=16
DATASET_DIRECTORY='painter-by-numbers/train'
DETECTOR_DIRECTORY='pretrained'


# Taken form https://stackoverflow.com/questions/57815001/pytorch-collate-fn-reject-sample-and-yield-another/57882783#57882783
def collate_fn_replace_corrupted(batch, dataset):
    # Idea from https://stackoverflow.com/a/57882783

    original_batch_len = len(batch)
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    filtered_batch_len = len(batch)
    # Num of corrupted examples
    diff = original_batch_len - filtered_batch_len
    if diff > 0:
        # Replace corrupted examples with another examples randomly
        batch.extend([dataset[random.randint(0, len(dataset)-1)] for _ in range(diff)])
        # Recursive call to replace the replacements if they are corrupted
        return collate_fn_replace_corrupted(batch, dataset)
    # Finally, when the whole batch is fine, return it
    return torch.utils.data.dataloader.default_collate(batch)


def main():
    t.backends.cudnn.benchmark=True

    t.manual_seed(RANDOM_SEED)

    if t.cuda.is_available():
        device = t.device("cuda")
    else:
        device = t.device("cpu")


    project_dir = os.getcwd()
    dataset_dir=os.path.join(project_dir, DATASET_DIRECTORY)
    detector_dir=os.path.join(project_dir, DETECTOR_DIRECTORY)

    transform = tv.transforms.Compose([tv.transforms.Normalize(mean=127.5,std=128),tv.transforms.ToTensor()])
    dataset = PainterByNumbers(dataset_dir,detector_dir,transform)

    custom_collate_fn = functools.partial(collate_fn_replace_corrupted, dataset=dataset)

    # loader = t.utils.data.DataLoader(dataset,batch_size=1,num_workers=1,collate_fn=custom_collate_fn)
    loader = t.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=cpu_count(),pin_memory=True,collate_fn=custom_collate_fn)






    # load the prerained head pose estimators
    fsanet1 = onnx.load(os.path.join(detector_dir, 'fsanet-1x1-iter-688590.onnx'))
    onnx.checker.check_model(fsanet1)
    fsanet2 = onnx.load(os.path.join(detector_dir, 'fsanet-var-iter-688590.onnx'))
    onnx.checker.check_model(fsanet2)

    # load the pretrained gender classifier
    resnet18 = onnx.load(os.path.join(detector_dir, 'resnet18-iter-AAAAA.onnx'))
    onnx.checker.check_model(resnet18)

    # prefer CUDA Execution Provider over CPU Execution Provider
    EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    # Start envs for running models in their onnx graph format
    fsanet1_session = onnxruntime.InferenceSession(fsanet1, providers=EP_list)
    fsanet2_session = onnxruntime.InferenceSession(fsanet2, providers=EP_list)
    resnet18_session = onnxruntime.InferenceSession(resnet18, providers=EP_list)


    metadata=[]

    with t.no_grad():
        for batch, filenames in loader:
            batch=batch.to(device)

            pose1 = fsanet1(batch)
            pose2 = fsanet2(batch)
            pose = t.mean(t.stack((pose1,pose2)),dim=0)

            # This is due to the FSA-Net implementation
            # yaw = pose[:,0]
            # pitch = pose[:,1]
            # roll = pose[:,2]

            logits = resnet(batch)
            genders = t.argmax(logits,dim=-1)

            # This is due to the Resnet18-based Gender Classifier implementation
            genders[genders==0] = 'female'
            genders[genders==1] = 'male'

            genders = genders.cpu().numpy()
            filenames = filenames.cpu().numpy()
            pose = pose.cpu().numpy()

            batch_metadata = {'filename':filenames,'yaw' : pose[:,0],'pitch' : pose[:,1], 'roll' : pose[:,2],'gender':genders}
            metadata.append(batch_metadata)

    df = pd.DataFrame(metadata)
    df.to_csv('paintings_metadata.csv',index=False)


if __name__ == '__main__':
    main()
