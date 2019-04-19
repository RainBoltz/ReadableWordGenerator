import os
import string
import numpy as np
from tqdm import tqdm, trange

class SGD_Utility:
    def __init__(self):
        self.lr_cnt = 10
        self.pos_lr_list = [1 + 0.1**x for x in range(5,5+self.lr_cnt)]
        self.neg_lr_list = [0.999**x for x in range(1,1+self.lr_cnt)]
        
    def negative_sampling_list(self, candidates, pos_index, samples=13):
        candidates.remove(pos_index)
        sample_list = np.random.choice(candidates, size=samples, replace=False)
        return sample_list
        
    def get_learning_rate(self, i_epoch):
        lr_index = int(np.floor(i_epoch/10)//self.lr_cnt)
        return self.pos_lr_list[lr_index], self.neg_lr_list[lr_index]

class BaseGenerator:
    def __init__(self):
        self.alphabet = list(string.ascii_lowercase)
        self.alphabet_length = len(self.alphabet)
        self.alphabet_index = { a:i for i,a in enumerate(self.alphabet) }
    
    def _get_words_property(self):
        with open(os.path.dirname(os.path.realpath(__file__))+'\\words.txt') as f:
            words = set(f.read().split())
        L = []
        for w in words:
            if len(w) not in L:
                L.append(len(w))
        return list(words), L
        

        

class WordGenerator_v1(BaseGenerator):
    def __init__(self):
        super().__init__()
        self.prob = np.random.rand(self.alphabet_length, self.alphabet_length)
    
    def train_generator(self, epochs, batch_size=None):  
        self.words, self.word_length = self._get_words_property()
        util = SGD_Utility()
        
        if batch_size == None:
            for e in trange(1,epochs+1, desc="training", ascii=True):
                np.random.shuffle(self.words)
                for w in self.words:
                    for i in range(len(w)-1):
                        this_alphabet = w[i]
                        next_alphabet = w[i+1]
                        pos_index = self.alphabet_index[next_alphabet]
                        neg_index_list = util.negative_sampling_list(list(range(self.alphabet_length)), pos_index)
                        pos_lr, neg_lr = util.get_learning_rate(e)
                        self.prob[self.alphabet_index[this_alphabet]][self.alphabet_index[next_alphabet]] *= pos_lr
                        for neg_index in neg_index_list:
                            self.prob[self.alphabet_index[this_alphabet]][neg_index] *= neg_lr
        else:
            for e in trange(1,epochs+1, desc="training", ascii=True):
                training_set = np.random.choice(self.words, size=batch_size, replace=False)
                for w in training_set:
                    for i in range(len(w)-1):
                        this_alphabet = w[i]
                        next_alphabet = w[i+1]
                        pos_index = self.alphabet_index[next_alphabet]
                        neg_index_list = util.negative_sampling_list(list(range(self.alphabet_length)), pos_index)
                        pos_lr, neg_lr = util.get_learning_rate(e)
                        self.prob[self.alphabet_index[this_alphabet]][self.alphabet_index[next_alphabet]] *= pos_lr
                        for neg_index in neg_index_list:
                            self.prob[self.alphabet_index[this_alphabet]][neg_index] *= neg_lr
        for a_i in range(self.alphabet_length):
            self.prob[a_i] = self.prob[a_i]/np.sum(self.prob[a_i])
                            
    def generate_word(self, word_length=None):
        if word_length == None:
            word_length = np.random.choice(self.word_length)
        else:
            if word_length not in self.word_length:
                word_length = np.random.choice(self.word_length)
        this_a_index = np.random.choice(list(range(self.alphabet_length)))
        the_word = self.alphabet[this_a_index]
        for i in range(word_length-1):
            next_a_index = np.random.choice(list(range(self.alphabet_length)),p=self.prob[this_a_index])
            the_word += self.alphabet[next_a_index]
            this_a_index = next_a_index
        return the_word

        

        
class WordGenerator_v2(BaseGenerator):
    def __init__(self):
        super().__init__()
        
    def train_generator(self, epochs, batch_size=None):    
        self.words, self.word_lengths = self._get_words_property()
        util = SGD_Utility()
        
        self.prob = { 
            l: np.random.rand(self.alphabet_length, self.alphabet_length) \
                for l in self.word_lengths
        }
        if batch_size == None:
            for e in trange(1,epochs+1, desc="training", ascii=True):
                np.random.shuffle(self.words)
                for w in self.words:
                    the_word_length = len(w)
                    for i in range(len(w)-1):
                        this_alphabet = w[i]
                        next_alphabet = w[i+1]
                        pos_index = self.alphabet_index[next_alphabet]
                        neg_index_list = util.negative_sampling_list(list(range(self.alphabet_length)), pos_index)
                        pos_lr, neg_lr = util.get_learning_rate(e)
                        self.prob[the_word_length][self.alphabet_index[this_alphabet]][self.alphabet_index[next_alphabet]] *= pos_lr
                        for neg_index in neg_index_list:
                            self.prob[the_word_length][self.alphabet_index[this_alphabet]][neg_index] *= neg_lr
        else:
            for e in trange(1,epochs+1, desc="training", ascii=True):
                training_set = np.random.choice(self.words, size=batch_size, replace=False)
                for w in training_set:
                    the_word_length = len(w)
                    for i in range(len(w)-1):
                        this_alphabet = w[i]
                        next_alphabet = w[i+1]
                        pos_index = self.alphabet_index[next_alphabet]
                        neg_index_list = util.negative_sampling_list(list(range(self.alphabet_length)), pos_index)
                        pos_lr, neg_lr = util.get_learning_rate(e)
                        self.prob[the_word_length][self.alphabet_index[this_alphabet]][self.alphabet_index[next_alphabet]] *= pos_lr
                        for neg_index in neg_index_list:
                            self.prob[the_word_length][self.alphabet_index[this_alphabet]][neg_index] *= neg_lr
        for w_l in self.word_length:
            for a_i in range(self.alphabet_length):
                self.prob[w_l][a_i] = self.prob[w_l][a_i]/np.sum(self.prob[w_l][a_i])
                
    def generate_word(self, word_length=None):
        if word_length == None:
            word_length = np.random.choice(self.word_length)
        else:
            if word_length not in self.word_length:
                word_length = np.random.choice(self.word_length)
        this_a_index = np.random.choice(list(range(self.alphabet_length)))
        the_word = self.alphabet[this_a_index]
        for i in range(word_length-1):
            next_a_index = np.random.choice(list(range(self.alphabet_length)),p=self.prob[word_length][this_a_index])
            the_word += self.alphabet[next_a_index]
            this_a_index = next_a_index
        return the_word
    

class WordGenerator_v3(BaseGenerator):
    def __init__(self):
        super().__init__()
        
    def train_generator(self, epochs, batch_size=None):    
        self.words, self.word_lengths = self._get_words_property()
        util = SGD_Utility()
        
        self.prob = {}
        for l in self.word_lengths:
            for _ in range(l):
                self.prob[l] = { 
                    i: np.random.rand(len(self.alphbet), len(self.alphbet)) \
                        for i in range(l-1)
                }
                
        if batch_size == None:
            for e in trange(1,epochs+1, desc="training", ascii=True):
                np.random.shuffle(self.words)
                for w in self.words:
                    the_word_length = len(w)
                    for i in range(len(w)-1):
                        this_alphabet = w[i]
                        next_alphabet = w[i+1]
                        pos_index = self.alphabet_index[next_alphabet]
                        neg_index_list = util.negative_sampling_list(list(range(self.alphabet_length)), pos_index)
                        pos_lr, neg_lr = util.get_learning_rate(e)
                        self.prob[the_word_length][i][self.alphabet_index[this_alphabet]][self.alphabet_index[next_alphabet]] *= pos_lr
                        for neg_index in neg_index_list:
                            self.prob[the_word_length][i][self.alphabet_index[this_alphabet]][neg_index] *= neg_lr
        else:
            for e in trange(1,epochs+1, desc="training", ascii=True):
                training_set = np.random.choice(self.words, size=batch_size, replace=False)
                for w in training_set:
                    the_word_length = len(w)
                    for i in range(len(w)-1):
                        this_alphabet = w[i]
                        next_alphabet = w[i+1]
                        pos_index = self.alphabet_index[next_alphabet]
                        neg_index_list = util.negative_sampling_list(list(range(self.alphabet_length)), pos_index)
                        pos_lr, neg_lr = util.get_learning_rate(e)
                        self.prob[the_word_length][i][self.alphabet_index[this_alphabet]][self.alphabet_index[next_alphabet]] *= pos_lr
                        for neg_index in neg_index_list:
                            self.prob[the_word_length][i][self.alphabet_index[this_alphabet]][neg_index] *= neg_lr
        for w_l in self.word_length:
            for w_l_i in range(w_l-1):
                for a_i in range(self.alphabet_length):
                    self.prob[w_l][w_l_i][a_i] = self.prob[w_l][w_l_i][a_i]/np.sum(self.prob[w_l][w_l_i][a_i])
    
    def generate_word(self, word_length=None):
        if word_length == None:
            word_length = np.random.choice(self.word_length)
        else:
            if word_length not in self.word_length:
                word_length = np.random.choice(self.word_length)
        this_a_index = np.random.choice(list(range(self.alphabet_length)))
        the_word = self.alphabet[this_a_index]
        for i in range(word_length-1):
            next_a_index = np.random.choice(list(range(self.alphabet_length)),p=self.prob[word_length][i][this_a_index])
            the_word += self.alphabet[next_a_index]
            this_a_index = next_a_index
        return the_word
    
    
    
    
    
    
    
if __name__ == "__main__":
    G = WordGenerator_v2()
    G.train_generator(epochs=200, batch_size=5000)
    print('---')
    while True:
        n = int(input('word length = '))
        print(G.generate_word(word_length=n))
    
    
    
    
    