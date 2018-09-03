# -*- coding: utf-8 -*-
"""
@author: Prabhu
"""

import numpy as np
from sklearn import metrics

def get_metrics(true_value, pred_probability, list_metrics):
    predicted = np.argmax(pred_probability, -1)
    output= {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(true_value, predicted)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(true_value, predicted)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrics' in list_metrics:
        output['confusion_matrics'] = metrics.confusion_matrix(true_value,predicted)
    return output



