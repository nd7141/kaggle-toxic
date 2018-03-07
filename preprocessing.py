import collections
import pandas as pd
from collections import defaultdict as ddict
from itertools import count
import numpy as np
import random, json

def build_dataset(comments):
    wcount = count()
    wdict = ddict(wcount.__next__)
    for comment in comments:
        words = comment.strip().split()
        for w in words:
            wdict[w]

    reverse_wdict = dict(zip(wdict.values(), wdict.keys()))
    return wdict, reverse_wdict, len(wdict)



def generate_batch(comments, reactions, N, wdict, batch_size, window_size):
    batch = np.ndarray(shape=(batch_size, 2*window_size + 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    R = reactions.shape[1]

    batch_row = 0
    for comments_row, comment in enumerate(comments):
        if sum(reactions[comments_row, :]) > 0:  # comment received at least one toxic value
            words = comment.strip().split()
            for target in range(len(words)):
                context = words[max((target - window_size), 0):target] + words[(target+1):min((target + window_size + 1), len(words))]
                context.extend([None]*(2*window_size - len(context))) # fill in dummy variables if None
                mapped_context = []
                for c in context:
                    if c is None:
                        mapped_context.append(N)
                    else:
                        mapped_context.append(wdict[c])
                # add reaction to batch

                for r_ix, r in enumerate(reactions[comments_row, :]):
                    if r > 0:
                        batch[batch_row, :] = mapped_context + [r_ix]
                        labels[batch_row, 0] = wdict[words[target]]
                        batch_row += 1
                        if batch_row == batch_size:
                            batch_row = 0
                            yield batch, labels

            # else: # comment didn't receive any toxic value and therefore is normal
            #     batch[batch_row, :] = mapped_context + [R] # R is the label of normal comment
            #     labels[batch_row, 0] = wdict[words[target]]
            #     batch_row += 1
            #     if batch_row == batch_size:
            #         batch_row = 0
            #         yield batch, labels


def generate_random_batch(comments, reactions, N, wdict, batch_size, window_size):
    batch = np.ndarray(shape=(batch_size, 2 * window_size + 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    R = reactions.shape[1]

    for batch_row in range(batch_size):
        comments_row = random.choice(list(range(len(comments))))
        comment = comments[comments_row]

        words = comment.strip().split()
        target = random.choice(list(range(len(words))))
        context = words[max((target - window_size), 0):target] + words[(target + 1):min((target + window_size + 1),
                                                                                        len(words))]

        context.extend([None] * (2 * window_size - len(context)))  # fill in dummy variables if None
        mapped_context = []
        for c in context:
            if c is None:
                mapped_context.append(N)
            else:
                mapped_context.append(wdict[c])

        if sum(reactions[comments_row, :]) > 0:
            indices = np.argwhere(reactions[comments_row, :])
            r_ix = random.choice(indices)
            batch[batch_row, :] = mapped_context + [r_ix]
            labels[batch_row, 0] = wdict[words[target]]
        else:
            batch[batch_row, :] = mapped_context + [R]
            labels[batch_row, 0] = wdict[words[target]]

    return batch, labels


if __name__ == '__main__':

    df = pd.read_csv('data/train_clean.csv')
    comments = df['comment_text']
    reactions = df.iloc[:, 2:].as_matrix()
    wdict, rev_wdict, N = build_dataset(comments)
    with open('wdict.json', 'w+') as f:
        json.dump(wdict, f)
    batch_size = 50
    window_size = 3

    # gen = generate_batch(comments, reactions, N, wdict, batch_size, window_size)
    # gen = generate_random_batch(comments, reactions, N, wdict, batch_size, window_size)
    # print(gen)


console = []