# -*- coding: utf-8 -*-
"""
@author: Prabhu
"""

import numpy as np
from sklearn import metrics

def get_metrics(true_value, predicted, list_metrics):
    # predicted = np.argmax(pred_probability, -1)
    output= {}
    if 'Accuracy' in list_metrics:
        output['Accuracy'] = metrics.accuracy_score(true_value, predicted)
    if 'Loss' in list_metrics:
        try:
            output['Loss'] = metrics.log_loss(true_value, predicted)
        except ValueError:
            output['Loss'] = -1
    if 'Confusion_matrics' in list_metrics:
        output['Confusion_matrix'] = metrics.confusion_matrix(true_value,predicted)

    with open('src/Result', 'w') as f:
        f.write(output)
    return output

