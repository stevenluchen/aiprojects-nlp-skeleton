import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class StartingDataset(torch.utils.data.Dataset):
    """
    Bag of Words Dataset
    """

    # TODO: dataset constructor.
    def __init__(self, data_path, is_train = True):
        '''
        data_path (str): path for the csv file that contains the data that you want to use
        '''
        # Preprocess the data. These are just library function calls so it's here for you
        self.df = pd.read_csv(data_path)
        if is_train:
            self.df = self.df.sample(frac = 0.9)
        self.vectorizer = CountVectorizer(stop_words = 'english', max_df = 0.99, min_df = 0.005)
        self.sequences = self.vectorizer.fit_transform(self.df.question_text.tolist()) # matrix of word counts for each sample
        self.labels = self.df.target.tolist() # list of labels
        self.token2idx = self.vectorizer.vocabulary_ # dictionary converting words to their counts
        self.idx2token = {idx: token for token, idx in self.token2idx.items()} # same dictionary backwards

    def __getitem__(self, i):
        '''
        i (int): the desired instance of the dataset
        '''
        # return the ith sample's list of word counts and label
        return torch.as_tensor(self.sequences[i, :].toarray()), self.labels[i]

    def __len__(self):
        return self.sequences.shape[0]