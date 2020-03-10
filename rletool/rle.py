import numpy as np
from itertools import groupby


class RLETool(object):
    # coco rle: [272, 2, 4, 4, 2, ...]
    def __init__(self, counts, size, mode='coco'):
        if mode not in ('coco', 'kaggle'):
            raise ValueError("mode should be 'coco' or 'kaggle'")

        self.counts = counts
        self.size = size
        self.mode = mode

        self.rle = {
            'counts': counts,
            'size': size
        }

    def decode(self):
        counts = self.counts
        shape = self.size[::-1]

        if self.mode == 'coco':
            bimask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
            n = 0
            val = 1
            for pos in range(len(counts)):
                val = not val
                for c in range(counts[pos]):
                    bimask[n] = val
                    n += 1
            #
            bimask = bimask.reshape(([shape[0], shape[1]]), order='F')
            #
            return bimask

        if self.mode == 'kaggle':
            bimask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
            starts, lengths = [np.asarray(x, dtype=int) for x in (counts[0:][::2], counts[1:][::2])]
            starts -= 1
            ends = starts + lengths
            for lo, hi in zip(starts, ends):
                bimask[lo:hi] = 1
            #
            bimask = bimask.reshape(shape, order='F')
            #
            return bimask

    @staticmethod
    def encode(bimask, mode):
        if not isinstance(bimask, np.ndarray):
            raise ValueError("bimask should be numpy.ndarray")

        if bimask.ndim != 2:
            raise ValueError("bimask should be 2 dimensions")

        if mode not in ('coco', 'kaggle'):
            raise ValueError("mode should be 'coco' or 'kaggle'")

        size = bimask.shape[::-1]

        if mode == 'coco':
            counts = list()
            for i, (value, elements) in enumerate(groupby(bimask.ravel(order='F'))):
                if i == 0 and value == 1:
                    counts.append(0)
                counts.append(len(list(elements)))
            #
            return RLETool(counts, size, mode)
        #
        if mode == 'kaggle':
            pixels = bimask.flatten()
            pixels = np.concatenate([[0], pixels, [0]])
            counts = np.where(pixels[1:] != pixels[:-1])[0] + 1
            counts[1::2] -= counts[::2]
            counts = counts.tolist()
            #
            return RLETool(counts, size, mode)

    def convert(self, mode):
        if mode not in ("coco", "kaggle"):
            raise ValueError("mode should be 'coco' or 'kaggle'")
        if mode == self.mode:
            return self
        #
        bimask = self.decode()

        return self.encode(bimask, mode)
