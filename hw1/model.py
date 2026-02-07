import torch
import torch.nn as nn
import zipfile
import numpy as np


class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path, weights_only=False)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    emb_matrix = np.random.uniform(-0.08, 0.08, (len(vocab), emb_size))

    print("Loading CloVe embedding...")

    with open(emb_file, "r", encoding = "utfg-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != emb_size + 1:
                continue
            word = parts[0]
            try:
                vec = np.array(parts[1:], dtype=np.float32)
            except ValueError:
                continue
            if word in vocab.word2id:
                idx = vocab.word2id[word]
                emb_matrix[idx] = vec
    print("Loaded")
    
    return emb_matrix




class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        vocab_size = len(self.vocab)
        emb_size = 300
        self.embedding = nn.Embedding(vocab_size, emb_size)
        #print(emb_size)
        self.fc1 = nn.Linear(emb_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, self.tag_size)

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        v = 0.08
        for p in self.parameters():
            nn.init.uniform_(p, -v, v)

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        emb_matrix = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        embeds = self.embedding(x) 
        
        
        if self.training and self.args.word_drop > 0:
            mask = torch.rand(embeds.size(0), embeds.size(1), 1, device=embeds.device)
            mask = (mask > self.args.word_drop).float()
            embeds = embeds * mask
            embeds = embeds / (1 - self.args.word_drop)  # scale to preserve magnitude
        
        
        h = embeds.mean(dim=1)
        h = torch.nn.functional.relu(self.fc1(h)) 
        h = torch.nn.functional.dropout(h, p=self.args.hid_drop, training=self.training) 
        h = torch.nn.functional.relu(self.fc2(h))
        h = torch.nn.functional.dropout(h, p=self.args.hid_drop, training=self.training) 
        h = torch.nn.functional.relu(self.fc3(h))
        h = torch.nn.functional.dropout(h, p=self.args.hid_drop, training=self.training) 
        scores = self.out(h)
        return scores
