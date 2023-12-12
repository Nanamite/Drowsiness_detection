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

    cf_matrix_normalized = cf_matrix/np.sum(cf_matrix, axis= 1)

    df_matrix = pd.DataFrame(cf_matrix_normalized, index= [0, 1], columns = [0, 1])

    val_acc = np.sum([cf_matrix_normalized[i, i] for i in range(2)])/2
    print("average test_acc: ", val_acc)

    fig = plt.figure()
    sn.heatmap(df_matrix, annot= True)
    plt.title('test results')
    plt.ylabel("true label")
    plt.xlabel("predicted label")

    return val_acc, fig, cf_matrix

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    model = gen_squeeze_net(False).to(device)
    model.load_state_dict(torch.load(r'squeeze_net_best_weights\best_model.pth'))

    _, _, cf = validate(model, device)

    tp = np.diag(cf)
    fp = np.sum(cf, axis = 0) - tp
    fn = np.sum(cf, axis = 1) - tp

    precision = np.mean(tp/(tp + fp))
    recall = np.mean(tp/(tp + fn))

    print("precision and recall: ", precision, ', ', recall)



