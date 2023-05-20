import torch as t
import torchvision as tv
import os
from multiprocessing import cpu_count


RANDOM_SEED=0
BATCH_SIZE=16
EPOCHS=5
TRAINING_DATASET_DIRECTORY='gender-classification-dataset/Training'
TESTING_DATASET_DIRECTORY='gender-classification-dataset/Validation'
DETECTOR_DIRECTORY='pretrained'


def main():
    t.backends.cudnn.benchmark=True

    t.manual_seed(RANDOM_SEED)

    if t.cuda.is_available():
        device = t.device("cuda")
    else:
        device = t.device("cpu")

    # The pretrained FSA-Net input size is 64x64
    transforms_train = tv.transforms.Compose([tv.transforms.Resize((64, 64)),tv.transforms.RandomHorizontalFlip(), tv.transforms.ToTensor(),tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transforms_val = tv.transforms.Compose([tv.transforms.Resize((64, 64)),tv.transforms.ToTensor(),tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    project_dir = os.getcwd()
    training_dataset = tv.datasets.ImageFolder(os.path.join(project_dir,TRAINING_DATASET_DIRECTORY))

    # Load a pretrained Resnet18 model and change its last fully connected layer such that it has 2 logits
    resnet = tv.models.resnet18(weights='IMAGENET1K_V1')
    num_features = resnet.fc.in_features
    resnet.fc = t.nn.Linear(num_features, 2)


if __name__ == '__main__':
    main()
