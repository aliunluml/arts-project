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
        width = painting.shape[1]
        height = painting.shape[0]

        # https://github.com/opencv/opencv/blob/3.4.0/samples/dnn/resnet_ssd_face_python.py
        detector_config_path = os.path.join(self.root_dir_path ,'pretrained','resnet10_ssd.prototxt')
        # https://github.com/opencv/opencv_3rdparty/tree/dnn_samples_face_detector_20170830
        detector_model_path = os.path.join(self.root_dir_path ,'pretrained','res10_300x300_ssd_iter_140000.caffemodel')

        conf_thres=0.7
        face_detector = cv2.dnn.readNetFromCaffe(detector_config_path, detector_model_path)
        # Resizing and standardizing the paintings dataset.
        # mean RGB=(104.0, 177.0, 123.0) is from the dataset used for training the face detector. We assume the paintings dataset is sharing the same mean with the training dataset of the face detector; it is assumed to be not OOD.
        face_detector.setInput(cv2.dnn.blobFromImage(painting, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False))
        detections = face_detector.forward()



        if(len(detections)>0):
            # If there are faces detected

            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            for i in range(0, detections.shape[2]):

                score = detections[0, 0, i, 2]

                # ensure that the detection greater than our threshold is
                # selected
                if score > conf_thres:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    box = box.astype("int")
                    (start_x, start_y, end_x, end_y) = box

                    # extract the face ROI and grab the ROI dimensions
                    face = painting[start_y:end_y, start_x:end_x]

                    (fh, fw) = face.shape[:2]
                    # ensure the face width and height are sufficiently large
                    if fw < 20 or fh < 20:
                        pass
                    else:
                        faces_bb.append(box)

        else:
            # Collate function in the dataloader deals with the portraits where faces are undetected
            return None
