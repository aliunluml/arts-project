# arts-project

Computational analysis of portraits.

Datasets: [Painter by numbers competition dataset from Kaggle](https://www.kaggle.com/competitions/painter-by-numbers/data) and [Gender classification dataset from Kaggle](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset).

Pipeline description: Pretrained Resnet10SSD Face Detector ([OpenCV Tutorial](https://github.com/opencv/opencv/blob/3.4.0/samples/dnn/resnet_ssd_face_python.py)), Ensemble of Pretrained FSA-Net Models with Different Scoring Functions ([Third party Pytorch implementation](https://github.com/omasaht/headpose-fsanet-pytorch)), and Resnet18-based Gender Classifier ([Third party Pytoch implementation](https://github.com/ndb796/Face-Gender-Classification-PyTorch))

Local development environment used is specified in the requirements.txt file
