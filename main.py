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
DATASET_DIRECTORY='train'
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
    return t.utils.data.dataloader.default_collate(batch)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

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

    transform = tv.transforms.Compose([tv.transforms.Normalize(mean=127.5,std=128)])
    dataset = PainterByNumbers(dataset_dir,detector_dir,transform)

    custom_collate_fn = functools.partial(collate_fn_replace_corrupted, dataset=dataset)

    # loader = t.utils.data.DataLoader(dataset,batch_size=1,num_workers=1,collate_fn=custom_collate_fn)
    loader = t.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=cpu_count(),pin_memory=True,collate_fn=custom_collate_fn)


    # the prerained head pose estimators
    fsanet1_path = os.path.join(detector_dir, 'fsanet-1x1-iter-688590.onnx')
    fsanet2_path = os.path.join(detector_dir, 'fsanet-var-iter-688590.onnx')

    # the pretrained gender classifier
    resnet18_path = os.path.join(detector_dir, 'resnet18-iter-14695.onnx')

    # prefer CUDA Execution Provider over CPU Execution Provider
    EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    # Start envs for running models in their onnx graph format
    fsanet1_session = onnxruntime.InferenceSession(fsanet1_path, providers=EP_list)
    fsanet2_session = onnxruntime.InferenceSession(fsanet2_path, providers=EP_list)
    resnet18_session = onnxruntime.InferenceSession(resnet18_path, providers=EP_list)

    metadata=[]

    with t.no_grad():
        for batch, filenames in loader:
            # Need to do this to have the correct memory ordering with the tensor elems. This is important because we pass this on as the buffer pointer to onnx input
            batch=batch.to(device).contiguous()

            # HEAD POSE ESTIMATION
            head_poses=[]

            # FSA-Net ensemble with a variance score function and a conv score function
            for fsanet_session in [fsanet1_session,fsanet2_session]:
                # print(fsanet_session.get_providers()) If CUDAExecutionProvider is absent, install onnxruntime-gpu and export PATHs to the local cuda executable
                # print(device)

                fsanet_session_binding = fsanet_session.io_binding()

                fsanet_session_binding.bind_input(name='input',device_type=device.type,device_id=0,element_type=np.float32,shape=tuple(batch.shape),buffer_ptr=batch.data_ptr())

                ## Allocate the PyTorch tensor for the model output
                fsanet_output_shape = (len(batch),3) # You need to specify the output PyTorch tensor shape
                fsanet_output = t.empty(fsanet_output_shape, dtype=t.float32).to(device).contiguous()
                fsanet_session_binding.bind_output(name='output',device_type=device.type,device_id=0,element_type=np.float32,shape=tuple(fsanet_output.shape),buffer_ptr=fsanet_output.data_ptr())

                fsanet_session.run_with_iobinding(fsanet_session_binding)

                head_poses.append(fsanet_output)

            avg_head_pose=t.mean(t.stack(head_poses),dim=0).cpu().numpy()
            yaw = avg_head_pose[:,0]
            pitch = avg_head_pose[:,1]
            roll = avg_head_pose[:,2]

            # GENDER CLASSIFICATION
            resnet18_session_binding = resnet18_session.io_binding()

            resnet18_session_binding.bind_input(name='input',device_type=device.type,device_id=0,element_type=np.float32,shape=tuple(batch.shape),buffer_ptr=batch.data_ptr())

            ## Allocate the PyTorch tensor for the model output
            resnet18_output_shape = (len(batch),2) # You need to specify the output PyTorch tensor shape
            resnet18_output = t.empty(resnet18_output_shape, dtype=t.float32).to(device).contiguous()
            resnet18_session_binding.bind_output(name='output',device_type=device.type,device_id=0,element_type=np.float32,shape=tuple(resnet18_output.shape),buffer_ptr=resnet18_output.data_ptr())

            resnet18_session.run_with_iobinding(resnet18_session_binding)

            logits=t.argmax(resnet18_output,dim=1).cpu().numpy()
            # Binary logits {0,1} correspond to 'Female' and 'Male' as per the alphabetical order in ImageFolder
            choices = ['F', 'M']
            gender=np.choose(logits, choices)

            # APPEND THE INFO TO SAVE LATER ON AS A CSV FILE
            batch_metadata = {'filename':filenames,'yaw' : yaw,'pitch' : pitch, 'roll' : roll,'gender':gender}
            metadata.append(batch_metadata)

    df = pd.DataFrame(metadata)
    df.to_csv('paintings_metadata.csv',index=False)

    all_data_info_df=pd.read_csv('all_data_info.csv')
    df=df.join(all_data_info_df.set_index('new_filename'), on='filename',how='inner')
    # Please select the columns needed fom all_data_info. This does not do copy()
    df=df[['date','gender','style','roll','yaw','pitch']]
    df.to_csv('paintings_all_data_info.csv',index=False)


if __name__ == '__main__':
    main()
