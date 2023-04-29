import torch as t
import pandas as pd
import os
import cv2


class PainterByNumbers(t.util.data.Dataset):
    def __init__(self,root_dir,transform=None:

        # root_dir = PATH for the project directory
        # transform = torchvision transformations
        csv_file_path=os.path.join(root_dir,'train_info.csv')
        self.train_info_df = pd.read_csv(csv_file_path)
        self.root_dir_path = root_dir
        self.transform=transform

    def __len__(self):
        result = len(self.train_info_df.index)
        return result

    def __getitem__(self,idx):
        file_path = os.path.join(self.root_dir_path ,'train'  ,self.train_info_df['filename'].iloc[idx])
        painting = cv2.imread(file_path)

        detector_config_path = os.path.join(self.root_dir_path ,'pretrained','resnet10_ssd.prototxt')
        detector_model_path = os.path.join(self.root_dir_path ,'pretrained','res10_300x300_ssd_iter_140000.caffemodel')

        conf_thres=0.7
        face_detector = cv2.dnn.readNetFromCaffe(detector_config_path, detector_model_path)

        face_detector.setInput(cv2.dnn.blobFromImage(painting, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False))
        detections = face_detector.forward()

        if(len(detections)>0):
            # If there are faces detected

        else:
            # Collate function in the dataloader deals with the portraits where faces are undetected
            return None
