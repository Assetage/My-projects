import numpy as np
import re
import pandas as pd

class Dataset:
    def __init__(self, X, y):
        self._x = X # сообщения 
        self._y = y # метки ["spam", "ham"]
        self.train = None # кортеж из (X_train, y_train)
        self.val = None # кортеж из (X_val, y_val)
        self.test = None # кортеж из (X_test, y_test)
        self.label2num = {"0":"ham","1":"spam"} # словарь, используемый для преобразования меток в числа
        self.num2label = {"ham":"0","spam":"1"} # словарь, используемый для преобразования числа в метки
        self._transform()
        
    def __len__(self):
        return len(self._x)
    
    def _transform(self):
        '''
        Функция очистки сообщения и преобразования меток в числа.
        '''
        # Начало вашего кода
        self._x = pd.Series(self._x)
        self._x = self._x.apply(lambda x: re.sub(r"\W+"," ",x))
        self._x = self._x.apply(lambda x: x.lower())
        self._x = self._x.apply(lambda x: x.strip())
        self._x = np.array(self._x)
        
        self._y = pd.Series(self._y)
        self._y = self._y.apply(lambda x: x.replace("ham", "0"))
        self._y = self._y.apply(lambda x: x.replace("spam", "1"))                       
        self._y = np.array(self._y)
                
        # Конец вашего кода
        pass

    def split_dataset(self, val=0.1, test=0.1):
        '''
        Функция, которая разбивает набор данных на наборы train-validation-test.
        '''
        # Начало вашего кода
        indices = np.arange(0, len(self._y))
        #np.random.seed(1)
        np.random.shuffle(indices)
        val_indices = indices[:round(val*len(self._y))]
        test_indices = indices[round(val*len(self._y)):round((val+test)*len(self._y))]
        train_indices = indices[round((val+test)*len(self._y)):]
        X_val = self._x[val_indices]
        y_val = self._y[val_indices]
        X_test = self._x[test_indices]
        y_test = self._y[test_indices]
        X_train = self._x[train_indices]
        y_train = self._y[train_indices]
        self.train = (X_train,y_train)
        self.val = (X_val,y_val)
        self.test = (X_test,y_test)

        # Конец вашего кода
        pass
