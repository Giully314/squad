import torch
from torch.utils.data import Dataset
import numpy as np

# taken from https://github.com/minggg/squad/blob/master/util.py
class SQuADDataset(Dataset):
    def __init__(self, filename: str):
        data = np.load(filename)
        self.context_idxs = torch.from_numpy(data['context_idxs'])
        self.context_char_idxs = torch.from_numpy(data['context_char_idxs'])
        self.question_idxs = torch.from_numpy(data['ques_idxs'])
        self.question_char_idxs = torch.from_numpy(data['ques_char_idxs'])
        self.y1s = torch.from_numpy(data['y1s'])
        self.y2s = torch.from_numpy(data['y2s'])
        self.ids = torch.from_numpy(data['ids'])    

        # SQuAD 2.0: Use index 0 for no-answer token (token 1 = OOV)
        batch_size, c_len, w_len = self.context_char_idxs.size()
        ones = torch.ones((batch_size, 1), dtype=torch.int32)
        self.context_idxs = torch.cat((ones, self.context_idxs), dim=1)
        self.question_idxs = torch.cat((ones, self.question_idxs), dim=1)
        ones = torch.ones((batch_size, 1, w_len), dtype=torch.int32)        
        self.context_char_idxs = torch.cat((ones, self.context_char_idxs), dim=1)
        self.question_char_idxs = torch.cat((ones, self.question_char_idxs), dim=1)
        self.y1s += 1
        self.y2s += 1

        self.valid_idxs = [idx for idx in range(len(self.ids))]

    def __getitem__(self, index: int) -> tuple:
        idx = self.valid_idxs[index]
        example = (self.context_idxs[idx],
                   self.context_char_idxs[idx],
                   self.question_idxs[idx],
                   self.question_char_idxs[idx],
                   self.y1s[idx],
                   self.y2s[idx],
                   self.ids[idx])
        return example


    def __len__(self) -> int:
        return len(self.valid_idxs)
    


# taken from https://github.com/minggg/squad/blob/master/util.py
# this function is used only with BiDAF model.
def bidaf_collate_fn(examples):
    """Create batch tensors from a list of individual examples returned
    by `SQuAD.__getitem__`. Merge examples of different length by padding
    all examples to the maximum length in the batch.

    Args:
        examples (list): List of tuples of the form (context_idxs, context_char_idxs,
        question_idxs, question_char_idxs, y1s, y2s, ids).

    Returns:
        examples (tuple): Tuple of tensors (context_idxs, context_char_idxs, question_idxs,
        question_char_idxs, y1s, y2s, ids). All of shape (batch_size, ...), where
        the remaining dimensions are the maximum length of examples in the input.

    Adapted from:
        https://github.com/yunjey/seq2seq-dataloader
    """
    def merge_0d(scalars, dtype=torch.int64):
        return torch.tensor(scalars, dtype=dtype)

    def merge_1d(arrays, dtype=torch.int64, pad_value=0):
        lengths = [(a != pad_value).sum() for a in arrays]
        padded = torch.zeros(len(arrays), max(lengths), dtype=dtype)
        for i, seq in enumerate(arrays):
            end = lengths[i]
            padded[i, :end] = seq[:end]
        return padded

    def merge_2d(matrices, dtype=torch.int64, pad_value=0):
        heights = [(m.sum(1) != pad_value).sum() for m in matrices]
        widths = [(m.sum(0) != pad_value).sum() for m in matrices]
        padded = torch.zeros(len(matrices), max(heights), max(widths), dtype=dtype)
        for i, seq in enumerate(matrices):
            height, width = heights[i], widths[i]
            padded[i, :height, :width] = seq[:height, :width]
        return padded

    # Group by tensor type
    context_idxs, context_char_idxs, \
        question_idxs, question_char_idxs, \
        y1s, y2s, ids = zip(*examples)

    # Merge into batch tensors
    context_idxs = merge_1d(context_idxs)
    context_char_idxs = merge_2d(context_char_idxs)
    question_idxs = merge_1d(question_idxs)
    question_char_idxs = merge_2d(question_char_idxs)
    y1s = merge_0d(y1s)
    y2s = merge_0d(y2s)
    ids = merge_0d(ids)

    return (context_idxs, context_char_idxs,
            question_idxs, question_char_idxs,
            y1s, y2s, ids)