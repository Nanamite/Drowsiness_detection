from prepare_data import prep
from generate_model import *
from sklearn.metrics import confusion_matrix
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn



def validate(model, device):
    _, _, test_loader = prep(1, True)

    preds = []
    targets = []

    model.eval()
    for idx, data in enumerate(test_loader):
        batch, labels = data
        batch = batch.to(device)
        labels = labels.to(device)

        outputs = model(batch)

        pred = torch.argmax(outputs).item()
        labels = labels.cpu().numpy()

        preds.append(pred)
        targets.append(labels)

    cf_matrix = confusion_matrix(targets, preds)

    cf_matrix = cf_matrix/np.sum(cf_matrix, axis= 1)

    df_matrix = pd.DataFrame(cf_matrix, index= [0, 1], columns = [0, 1])

    val_acc = np.sum([cf_matrix[i, i] for i in range(2)])/2
    print("average test_acc: ", val_acc)

    fig = plt.figure()
    sn.heatmap(df_matrix, annot= True)
    plt.title('test results')
    plt.ylabel("true label")
    plt.xlabel("predicted label")

    return val_acc, fig

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    model = gen_mobile_net(False).to(device)
    model.load_state_dict(torch.load(r'mobile_net_saves_new\2\best_model.pth'))

    validate(model, device)



