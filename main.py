import functools
import random
import torch as t
import os
from preprocessing import PainterByNumbers


# Taken form https://stackoverflow.com/questions/57815001/pytorch-collate-fn-reject-sample-and-yield-another/57882783#57882783
def collate_fn_replace_corrupted(batch, dataset):
    """Collate function that allows to replace corrupted examples in the
    dataloader. It expect that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are replaced with another examples sampled randomly.

    Args:
        batch (torch.Tensor): batch from the DataLoader.
        dataset (torch.utils.data.Dataset): dataset which the DataLoader is loading.
            Specify it with functools.partial and pass the resulting partial function that only
            requires 'batch' argument to DataLoader's 'collate_fn' option.

    Returns:
        torch.Tensor: batch with new examples instead of corrupted ones.
    """
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
    project_dir = os.getcwd()
    dataset_dir=os.path.join(project_dir, 'train_4')
    detector_dir=os.path.join(project_dir, 'pretrained')
    dataset = PainterByNumbers(dataset_dir,detector_dir)

    custom_collate_fn = functools.partial(collate_fn_replace_corrupted, dataset=dataset)
    loader = t.utils.data.DataLoader(dataset,batch_size=1,num_workers=1,collate_fn=custom_collate_fn)
    # loader = DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,collate_fn=custom_collate_fn)

    inputs=next(iter(loader))

if __name__ == '__main__':
    main()
