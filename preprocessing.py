import torch as t
import numpy as np
import os
import cv2
import sys



class PainterByNumbers(t.utils.data.Dataset):
    def __init__(self,dataset_dir,detector_dir,transform=lambda x:x):

        self.transform=transform
        self.dataset_path=dataset_dir
        self.detector_path=detector_dir

    def __len__(self):
        filenames=os.listdir(self.dataset_path)
        result = len(filenames)
        return result

    # private method
    def __getroi__(self,image,detector,thres=0.7):
        width = image.shape[1]
        height = image.shape[0]

        # Resizing and standardizing the paintings dataset.
        # mean RGB=(104.0, 177.0, 123.0) is from the dataset used for training the face detector. We assume the paintings dataset is sharing the same mean with the training dataset of the face detector; it is assumed to be not OOD.
        # Convert input image to NCHW order [batchsize,channels,height,width] and make a blob
        input=cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

        # OpenCV NNs do not support having an input batch, which includes multiple blobs. We would need to have a Pytorch NN outside the dataset to work efficiently on minibatches in the future. For now, this is what it is.
        detector.setInput(input)

        detections = detector.forward()

        batch_size,channels,topk,bb_data_size=detections.shape
        # batch_size is either 0 or 1 because we input a single blob.
        if(batch_size>0):
            # If there are faces detected
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            # topk BB detections are in the descending order wrt their confidence scores
            for i in range(0,topk):
                # Normally, we would need to check im_index_in_batch but we don't as we have only a single image
                # im_index_in_batch = detections[0, 0, i, 0]
                # class_label = detections[0, 0, i, 1]
                score = detections[0, 0, i, 2]

                # ensure that the detection greater than our threshold is
                # selected
                if score > thres:
                    # compute the (x, y)-coordinates of the bounding box for the original painting
                    # typecasting from numpy array float to integer

                    x1 = int(detections[0, 0, i, 3] * width)
                    y1 = int(detections[0, 0, i, 4] * height)
                    x2 = int(detections[0, 0, i, 5] * width)
                    y2 = int(detections[0, 0, i, 6] * height)

                    # extract the face ROI as bounding box and grab its dimensions
                    bb = image[y1:y2+1, x1:x2+1]
                    # painting is not in the NCHW order
                    (face_height, face_width) = bb.shape[:2]
                    # ensure the face width and height are sufficiently large. input image is 300x300
                    if (face_width < 20 or face_height < 20):
                        pass
                    else:
                        return bb
                else:
                    pass
            # Collate function in the dataloader deals with the bbs that are not sufficiently confident or large.
            return None

        else:
            # Collate function in the dataloader deals with the portraits where no face is detected
            return None


    def __getitem__(self,idx):
        # Filenames start from '1.jpg', not '0.jpg' because we rename partition datasets during development to be as such
        filename=str(idx+1)+'.jpg'
        file_path = os.path.join(self.dataset_path, filename)
        # OpenCV loads the image in BGR channel order
        painting = cv2.imread(file_path)

        # https://github.com/opencv/opencv/blob/3.4.0/samples/dnn/resnet_ssd_face_python.py
        detector_config_path = os.path.join(self.detector_path,'resnet10_ssd.prototxt')
        # https://github.com/opencv/opencv_3rdparty/tree/dnn_samples_face_detector_20170830
        detector_model_path = os.path.join(self.detector_path,'res10_300x300_ssd_iter_140000.caffemodel')

        face_detector = cv2.dnn.readNetFromCaffe(detector_config_path, detector_model_path)

        conf_thres=0.7

        face = self.__getroi__(painting,face_detector,conf_thres)

        if face is None:
            return None
        else:
            # resize for FSA-Net. It is already standardized.
            face = cv2.resize(face,(64,64))
            # OpenCV has BGR order whereas Pytorch has RGB. This does not impact the pretrained FSA-Net we have in our pipeline because it was trained on RGB images.
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # convert HWC format to CHW
            face = np.transpose(face,(2,0,1))
            # Apply data tranformations/augmentations/etc.
            x = self.transform(face)
            
            return (x, filename)
