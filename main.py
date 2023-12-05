import functools
import random
import torch as t
import numpy as np
import pandas as pd
import torchvision as tv
import cv2
import os
import sys
import onnx
import onnxruntime
from dataset import PainterByNumbers
from multiprocessing import cpu_count


RANDOM_SEED=0
BATCH_SIZE=16
DATASET_DIRECTORY='toy_train'
OUT=True
OUT_DIRECTORY='out'
DETECTOR_DIRECTORY='pretrained'
DATA_CSV_FILENAME='all_data_info.csv'


# Taken from https://github.com/omasaht/headpose-fsanet-pytorch/blob/master/src/utils.py
# Originally from https://github.com/shamangary/FSA-Net/blob/master/demo/demo_FSANET.py
def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 50,thickness=(2,2,2)):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),thickness[0])
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),thickness[1])
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),thickness[2])

    return img


# Taken form https://stackoverflow.com/questions/57815001/pytorch-collate-fn-reject-sample-and-yield-another/57882783#57882783
def collate_fn_replace_nonface(batch, dataset):
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
        return collate_fn_replace_nonface(batch, dataset)
    # Finally, when the whole batch is fine, return it
    return t.utils.data.dataloader.default_collate(batch)


# Taken from https://discuss.pytorch.org/t/questions-about-dataloader-and-dataset/806
def collate_fn_skip_nonface(batch):
    batch = list(filter(lambda x: x is not None, batch))
    # Sometimes all samples are nonface. Then, we return a None object instead of default_collate because the latter expects a non-empty batch
    if len(batch)!=0:
        return t.utils.data.dataloader.default_collate(batch)
    else:
        return (None, None, None, None)


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
    csv_file_path=os.path.join(project_dir, DATA_CSV_FILENAME)
    if OUT:
        out_dir=os.path.join(project_dir, OUT_DIRECTORY)
    else:
        out_dir=None

    # https://github.com/opencv/opencv/blob/3.4.0/samples/dnn/resnet_ssd_face_python.py
    detector_config_path = os.path.join(dataset_dir,'resnet10_ssd.prototxt')
    # https://github.com/opencv/opencv_3rdparty/tree/dnn_samples_face_detector_20170830
    detector_model_path = os.path.join(dataset_dir,'res10_300x300_ssd_iter_140000.caffemodel')

    face_detector = cv2.dnn.readNetFromCaffe(detector_config_path, detector_model_path)


    transform = tv.transforms.Compose([tv.transforms.Normalize(mean=127.5,std=128)])
    dataset = PainterByNumbers(dataset_dir,face_detector,csv_file_path,transform)

    custom_collate_fn = functools.partial(collate_fn_replace_nonface, dataset=dataset)

    # loader = t.utils.data.DataLoader(dataset,batch_size=1,num_workers=1,collate_fn=custom_collate_fn)
    # Use one worker process to load data. Otherwise, index errors sometimes pop up.
    loader = t.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=True,collate_fn=collate_fn_skip_nonface)


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
        for i,(batch, filenames, num_bbss, bbs) in enumerate(loader):
            print(i)
            # If there are no faces in the current batch, skip it
            if batch is None:
                continue
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

            avg_head_pose = t.mean(t.stack(head_poses),dim=0).cpu().numpy()
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

            logits = t.argmax(resnet18_output,dim=1).cpu().numpy()
            # Binary logits {0,1} correspond to 'Female' and 'Male' as per the alphabetical order in ImageFolder
            choices = ['F', 'M']
            gender = np.choose(logits, choices)

            # Convert a tensor object to a numpy float array because we do not want tensor objects in our CSV file. A numpy float is the same as a Python float.
            num_bbss=num_bbss.numpy()

            # Tuple is converted to a list
            # bbs={'upperLeft': [tensor([1...len(batch)]), tensor([1...len(batch)])], 'lowerRight': [tensor([1...len(batch)]), tensor([1...len(batch)])]}
            upper_left=bbs['upperLeft']
            lower_right=bbs['lowerRight']

            x1s=upper_left[0].numpy()
            y1s=upper_left[1].numpy()
            x2s=lower_right[0].numpy()
            y2s=lower_right[1].numpy()

            # APPEND THE INFO TO SAVE LATER ON AS A CSV FILE
            batch_metadata = [{'filename':f,'yaw' : y,'pitch' : p, 'roll' : r,'gender':g,'num_faces':num_bbs,'x1':x1,'y1':y1,'x2':x2,'y2':y2} for f,y,p,r,g,num_bbs,x1,y1,x2,y2 in zip(filenames, yaw, pitch, roll, gender, num_bbss, x1s, y1s, x2s, y2s)]
            metadata.extend(batch_metadata)

            if out_dir is not None:
                # Iterates through all batch items, even the duplicates. Not elegant.
                for i in range(0,len(batch)):
                    im=batch[i].cpu().numpy()
                    # denormalize to have pixel values in [0,255]
                    im=(im*128+127.5).astype(np.uint8)
                    # convert CHW format to HWC
                    im = np.transpose(im,(1,2,0))
                    # convert RGB channel coding to BGR, which is the default for Opencv
                    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                    # draw head pose arrows
                    im = draw_axis(im,yaw[i],pitch[i],roll[i])
                    # annotate the gender
                    x = im.shape[1]//2
                    y = im.shape[0]//2
                    im = cv2.putText(im,gender[i],(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
                    # save it
                    cv2.imwrite(os.path.join(out_dir, filenames[i]),im)
            else:
                pass



    df = pd.DataFrame(metadata)
    # Remove duplicate images that are resampled with collate_fn_replace_nonface
    df = df.drop_duplicates(subset='filename')
    # Save model outputs
    df.to_csv('paintings_metadata.csv',index=False)

    # Save model outputs combined with other related info from the paintings dataset
    all_data_info_df = pd.read_csv(csv_file_path)
    df = df.join(all_data_info_df.set_index('new_filename'), on='filename',how='inner')

    # Please select the columns needed fom all_data_info. This does not do copy()
    df = df[['artist','date','style','pixelsx','pixelsy','num_faces','roll','yaw','pitch','gender','x1','y1','x2','y2']]
    df.to_csv('paintings_all_data_info.csv',index=False)



if __name__ == '__main__':
    main()
