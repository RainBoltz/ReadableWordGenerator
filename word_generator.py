import os
import string
import numpy as np

class BaseGenerator:
    def __init__(self):
        self.alphabet = list(string.ascii_lowercase)
        self.alphabet_length = len(self.alphabet)
    
    def _get_words(self):
        with open(os.path.dirname(os.path.realpath(__file__))+'\\words.txt') as f:
            words = set(f.read().split())
        return words
        
    def _negative_sampling(self):
        

class WordGenerator_v1:
    def __init__(self):
        super.__init__()
        self.prob = np.random.rand(self.alphabet_length, self.alphabet_length)
    
    
        
    
    def train_generator(self, epochs):    
        self.words = self._get_words()

        

        
class WordGenerator_v2:
    def __init__(self):
        self.alphbet = list(string.ascii_lowercase)
        
    def train_generator(self):    
        self.words = self._get_words()
        L = []
        for w in words:
            if len(w) not in L:
                L.append(len(w))
        self.prob = { l: np.random.rand(self.alphabet_length, self.alphabet_length) for l in L }
    

class WordGenerator_v3:
    def __init__(self):
        self.alphbet = list(string.ascii_lowercase)
        
    def train_generator(self):    
        self.words = self._get_words()
        L = []
        for w in words:
            if len(w) not in L:
                L.append(len(w))
        self.prob = {}
        for l in L:
            for _ in range(l):
                self.prob[l] = { i: np.random.rand(len(self.alphbet), len(self.alphbet)) for i in range(l) }
    
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    
    
    
    
    
    
    