import numpy as np
import random
import pandas as pd
import json

class Predict(object):
    def __init__(self,
                 doc_embeddings,
                 word_embeddings,
                 wdict,
                 test_comments,
                 weights,
                 biases):
        self.doc_embeddings = doc_embeddings
        self.word_embeddings = word_embeddings
        self.wdict = wdict
        self.test_comments = test_comments
        self.weights = weights
        self.biases = biases

    def softmax(self, word_row):
        print('inside softmax')
        w_emb = self.weights[word_row, :]
        doc_embs = self.doc_embeddings
        b = self.biases[word_row]
        print(doc_embs.shape, w_emb.shape, b)
        soft = np.exp(doc_embs.dot(w_emb) + b)/np.sum(np.exp(doc_embs.dot(w_emb) + b))
        print(soft)
        return soft

    def softmax_context(self, target_word, context_words):
        if context_words:
            avg_contexts = np.mean(self.word_embeddings[context_words, :], axis=0)
            avg_contexts = avg_contexts.reshape(1, len(avg_contexts))
            A = np.repeat(avg_contexts, self.doc_embeddings.shape[0], axis=0)
            D = np.hstack((A, self.doc_embeddings))
            b = self.biases.reshape(self.biases.shape[0], 1)
            presoft = np.exp(self.weights.dot(D.T) + b)
            soft = presoft[target_word, :]/np.sum(presoft, axis=0)

            return soft

    def sampled_softmax_context(self, target_word, context_words):
        if context_words:
            avg_contexts = np.mean(self.word_embeddings[context_words, :], axis=0)
            avg_contexts = avg_contexts.reshape(1, len(avg_contexts))
            A = np.repeat(avg_contexts, self.doc_embeddings.shape[0], axis=0)
            D = np.hstack((A, self.doc_embeddings))
            b = self.biases.reshape(self.biases.shape[0], 1)
            num = np.exp(D.dot(self.weights[target_word, :]) + b[target_word])
            sampled = np.random.choice(list(range(self.weights.shape[0])), 30)
            denom = np.sum(np.exp(D.dot(self.weights[sampled, :].T) + b[sampled].reshape(1, len(b[sampled]))), axis=1)
            soft = num/denom

            return soft

    def predict_comment2(self, comment, window_size = 4, func = np.max):
        P = []
        for target in range(len(comment)):
            context = comment[max((target - window_size), 0):target] + \
                      comment[(target + 1):min((target + window_size + 1), len(comment))]
            context_words = []
            for c in context:
                if c in wdict:
                    context_words.append(wdict[c])
            if context_words:
                word = comment[target]
                if word in self.wdict:
                    word_row = self.wdict[word]
                    probs = self.sampled_softmax_context(word_row, context_words)
                    P.append(probs)
        print(P)
        if P:
            scores = func(P, axis=0)
        else:
            scores = [0.5] * self.doc_embeddings.shape[0]
        return scores

    def predict_comment(self, comment, func = np.max):
        P = []
        for word in comment:
            if word in self.wdict:
                word_row = self.wdict[word]
                probs = self.softmax(word_row)
                P.append(probs)
        if P:
            scores = func(P, axis = 0)
        else:
            scores = [0.5]*self.doc_embeddings.shape[0]
        return scores

    def predict(self, func = np.max):
        P = np.ndarray(shape = (len(self.test_comments), self.doc_embeddings.shape[0]))
        for test_row, line in enumerate(self.test_comments):
            if not test_row % 10000:
                print(test_row)
            # try:
            comment = line.strip().split()
            scores = self.predict_comment2(comment, func = func)
            print(test_row, scores)
            # except:
            #     scores = [0.5]*self.doc_embeddings.shape[0]
            P[test_row, :] = scores
        return P

if __name__ == '__main__':
    # Set random seeds
    SEED = 2018
    random.seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv('data/test_clean.csv')
    comments = df['comment_text']
    reactions = df.iloc[:, 2:].as_matrix()

    W = np.load('output/word_embeddings.npy')
    D = np.load('output/doc_embeddings.npy')
    weights = np.load('output/weights.npy')
    biases = np.load('output/biases.npy')

    with open('wdict.json') as f:
        wdict = json.load(f)

    model = Predict(D, W, wdict, comments, weights, biases)
    P = model.predict(func = np.mean)
    #
    # df['toxic'] = P[:, 0]
    # df['severe_toxic'] = P[:, 1]
    # df['obscene'] = P[:, 2]
    # df['threat'] = P[:, 3]
    # df['insult'] = P[:, 4]
    # df['identity_hate'] = P[:, 5]
    #
    # df.drop('comment_text', axis = 1, inplace=True)
    #
    # df.to_csv('data/test_predicted.csv', index=False, encoding='utf-8')


    console = []