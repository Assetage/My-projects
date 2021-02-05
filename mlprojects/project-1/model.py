import numpy as np
import re
from dataset import Dataset
import pandas as pd

class Model:
    def __init__(self, alpha=1):
        self.vocab = set() # словарь, содержащий все уникальные слова из набора train
        self.spam = {} # словарь, содержащий частоту слов в спам-сообщениях из набора данных train.
        self.ham = {} # словарь, содержащий частоту слов в не спам-сообщениях из набора данных train.
        self.alpha = alpha # сглаживание
        self.label2num = {"ham":"0","spam":"1"} # словарь, используемый для преобразования меток в числа
        self.num2label = {"0":"ham","1":"spam"} # словарь, используемый для преобразования числа в метки
        self.Nvoc = None # общее количество уникальных слов в наборе данных train
        self.Nspam = None # общее количество уникальных слов в спам-сообщениях в наборе данных train
        self.Nham = None # общее количество уникальных слов в не спам-сообщениях в наборе данных train
        self._train_X, self._train_y = None, None
        self._val_X, self._val_y = None, None
        self._test_X, self._test_y = None, None

    def fit(self, dataset):
        '''
        dataset - объект класса Dataset
        Функция использует входной аргумент "dataset", 
        чтобы заполнить все атрибуты данного класса.
        '''
        # Начало вашего кода
        self._train_X, self._train_y = dataset.train[0], dataset.train[1]
        self._val_X, self._val_y = dataset.val[0], dataset.val[1]
        self._test_X, self._test_y = dataset.test[0], dataset.test[1]
        
        for row in self._train_X:
            for word in row.split(" "):
                self.vocab.add(word)
        
        self.Nvoc = len(self.vocab)
        
        
        for row1, row2 in zip(self._train_X, self._train_y):    
            for word in row1.split(" "):        
                if word not in getattr(self,self.num2label[row2]):            
                    getattr(self,self.num2label[row2])[word] = 1
                else:
                    getattr(self,self.num2label[row2])[word] = getattr(self,self.num2label[row2])[word] + 1
                    
        self.Nspam = 0
        for i in self.spam:
            self.Nspam = self.Nspam + self.spam[i]
        
        self.Nham = 0
        for i in self.ham:
            self.Nham = self.Nham+ self.ham[i]      
        
        all_messages = len(self._train_X)
        spam_messages = 0
        ham_messages = 0
        for row in self._train_y:
            if row == "1":
                spam_messages = spam_messages + 1
            elif row == "0":
                ham_messages = ham_messages + 1       
        
        p_spam = spam_messages / all_messages
        p_ham = ham_messages / all_messages
             
        self.p_spam = p_spam
        self.p_ham = p_ham

        # Конец вашего кода
        pass
    
    def inference(self, message):
        '''
        Функция принимает одно сообщение и, используя наивный байесовский алгоритм, определяет его как спам / не спам.
        '''
        # Начало вашего кода
        message = re.sub(r"\W+"," ",message)
        message = message.lower()
        
        word_prob_spam = 1
        word_prob_ham = 1
        word_prob = []
        for Nsam,sam,word_prob_sam in zip([self.Nspam,self.Nham],[self.spam,self.ham],[word_prob_spam,word_prob_ham]):
            for word in message.split(" "):
                if word.lower() not in sam:
                    word_prob_sam = word_prob_sam * (self.alpha) / (Nsam + self.Nvoc * self.alpha)
                else:
                    word_prob_sam = word_prob_sam * (sam[word.lower()] + self.alpha) / (Nsam + self.Nvoc * self.alpha)
            word_prob.append(word_prob_sam) 

        pspam = self.p_spam * word_prob[0]     
        pham = self.p_ham * word_prob[1]
       
        # Конец вашего кода
        if pspam > pham:
            return "spam"
        return "ham"
    
    def validation(self):
        '''
        Функция предсказывает метки сообщений из набора данных validation,
        и возвращает точность предсказания меток сообщений.
        Вы должны использовать метод класса inference().
        '''
        # Начало вашего кода
        predicted_val = 0
        for row1,row2 in zip(self._val_X,self._val_y):
            if self.inference(row1) == self.num2label[row2]:
                predicted_val = predicted_val + 1
            else:
                predicted_val = predicted_val
        
        val_acc = predicted_val / len(self._val_y)
        val_acc = f"{round(val_acc * 100, 2)}%"
       
        # Конец вашего кода
        return val_acc 

    def test(self):
        '''
        Функция предсказывает метки сообщений из набора данных test,
        и возвращает точность предсказания меток сообщений.
        Вы должны использовать метод класса inference().
        '''
        # Начало вашего кода
        predicted_test = 0
        for row1,row2 in zip(self._test_X,self._test_y):
            if self.inference(row1) == self.num2label[row2]:
                predicted_test = predicted_test + 1
            else:
                predicted_test = predicted_test
        
        test_acc = predicted_test / len(self._test_y)
        test_acc = f"{round(test_acc * 100, 2)}%"
        
        # Конец вашего кода
        return test_acc


