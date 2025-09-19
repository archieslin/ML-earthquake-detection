#!/bin/python
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader
import obspy
import time
from obspy import read
from obspy import UTCDateTime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
start_time = time.time()
datapath = '/home/sysop/pytorch/Data/'



# 0. Import params
params = {}
with open('in.par2', 'r') as f:
    for line in f:
        if line.strip() and not line.strip().startswith('#'):
            key, value = line.strip().split('=', 1)
            key = key.strip()
            value = value.strip()

            if value.isdigit():
                params[key] = int(value)
            elif value.replace('.', '', 1).isdigit():
                params[key] = float(value)
            else:
                params[key] = value.strip('"').strip("'")


# 1. Import data (training & validation)
try:
    st = read(datapath + params["mseed"])
except Exception as e:
    print(f"Error loading waveform data: {e}")
    st = None

if st:
    t2 = obspy.UTCDateTime(params["t"])
    st_sliced = st.slice(t2, t2 + params["dt"])
    st2 = st_sliced.copy()

    st2.normalize()
    st2.detrend()

    processed_data = []
    labels = []

    all_processed_traces = [i.data for i in st2]

    X_test_data = all_processed_traces
    Y_test_labels = []
    for i in range(len(st2)):
        cond1 = st2[i].stats.station in params["label_test"]
        station_channel = f"{st2[i].stats.station}.{st2[i].stats.channel}"
        cond2 = station_channel in params["label_test2"]
        if cond1 or cond2:
            Y_test_labels.append(1)
            print(station_channel)
        else:
            Y_test_labels.append(0)

    # Y_test_labels = [1 if st2[i].stats.station in params["label_test"] else 0 for i in range(len(st2))]

    print("Waveform data loaded and processed with labels.")
    print(f"Number of testing data points: {len(X_test_data)}", "\n")

else:
    print("Failed to load waveform data.")


# 2. Model setting
class DynamicLengthRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(DynamicLengthRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out

model = DynamicLengthRNN(input_size=params["input_size"], hidden_size=params["hidden_size"], num_layers=params["num_layers"], num_classes=params["num_classes"])
model.load_state_dict(torch.load('1Drnn.pth'))
model.eval()
print(model)
print("model is been prepared", "\n")


# 3. Loading data
X_test_tensor = torch.tensor(np.array(X_test_data), dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test_labels, dtype=torch.long)
X_test_tensor = X_test_tensor.unsqueeze(1)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 4. Model
classified_as_00 = []
classified_as_11 = []
misclassified_as_01 = [] 
misclassified_as_10 = [] 
all_labels = []
all_predictions = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with torch.no_grad():
    inputs = X_test_tensor.to(device)
    labels = Y_test_tensor.to(device)

    outputs = model(inputs)

    _, predicted = torch.max(outputs.data, 1)


    all_labels = labels.cpu().numpy().tolist()
    all_predictions = predicted.cpu().numpy().tolist()


    for i in range(len(labels)):
        true_label = labels[i].item()
        predicted_label = predicted[i].item()


        original_trace_index = i
        station_name = st_sliced[original_trace_index].stats.station
        channel_name = st_sliced[original_trace_index].stats.channel

        if true_label == 1 and predicted_label == 0:
            misclassified_as_01.append((inputs[i], true_label, predicted_label, station_name, channel_name))

        elif true_label == 0 and predicted_label == 1:
            misclassified_as_10.append((inputs[i], true_label, predicted_label, station_name, channel_name))

        elif true_label == 0 and predicted_label == 0:
            classified_as_00.append((inputs[i], true_label, predicted_label, station_name, channel_name))

        elif true_label == 1 and predicted_label == 1:
            classified_as_11.append((inputs[i], true_label, predicted_label, station_name, channel_name))

    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    accuracy = 100 * correct / total

    print(f'Test data accuracy: {accuracy:.2f} %')



fig, axes = plt.subplots(4, 3, figsize=(15, 15))
fig.suptitle("Testing Examples - Model Performance Overview", fontsize=16)


categories = {
    "Label: 0, Predict: 0": {"examples": classified_as_00, "color": "#1f77b4"},
    "Label: 1, Predict: 1": {"examples": classified_as_11, "color": "#1f77b4"},
    "Label: 0, Predict: 1 (Misclassified)": {"examples": misclassified_as_10, "color": "#d62728"},
    "Label: 1, Predict: 0 (Misclassified)": {"examples": misclassified_as_01, "color": '#d62728'}
}

for i, (title, category_info) in enumerate(categories.items()):
    examples = category_info["examples"]
    color = category_info["color"]
    sampled_examples = random.sample(examples, min(len(examples), 3))

    for j in range(len(sampled_examples)):
        ax = axes[i, j]
        data, label, predict, station_name, channel_name = sampled_examples[j]
        new_title = f"Station: {station_name}, Channel: {channel_name}\n{title}"
        ax.plot(data.numpy()[0], color=color)
        ax.set_title(new_title, fontsize=10)
        ax.set_xticks(np.linspace(0, len(data[0]) - 1, 6))
        ax.set_xticklabels(np.linspace(0, 5, 6))
        ax.set_xlabel("Time (s)")

    for j in range(len(sampled_examples), 3):
        fig.delaxes(axes[i, j])


plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
plt.savefig("RNN_test_examples_overview_TEST.png", dpi=300)
print("[PNG]test_examples_overview be stored", "\n")



# 6. Confusion matrix
if not all_labels or not all_predictions:
    print("Warning: The label and prediction lists are empty. Please run the model evaluation first.")
else:
    all_labels_np = np.array(all_labels)
    all_predictions_np = np.array(all_predictions)
    total = len(all_labels_np)

    cm = confusion_matrix(all_labels_np, all_predictions_np)

    tn, fp, fn, tp = cm.ravel()

    correct_rate = (tp + tn) / total
    incorrect_rate = (fp + fn) / total

    print("\n--- Confusion matrix ---")
    print(f"Total: {total}")
    print(f"Correct: {tp + tn} ({correct_rate:.2%})")
    print(f"Miscorrect: {fp + fn} ({incorrect_rate:.2%})")
    print("-" * 30)
    print(f"True Positives (TP): {tp} (label 1, Pred 1)")
    print(f"True Negatives (TN): {tn} (label 0, Pred 0)")
    print(f"False Positives (FP): {fp} (label 0, Pred 1 - Mis)")
    print(f"False Negatives (FN): {fn} (label 1, Pred 0 - Omit)")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-earthquake (0)", "Earthquake (1)"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()
    plt.savefig("RNN_confusion_matrix_TEST.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    print("\n[PNG] confusion_matrix be stored")
    plt.close(fig)


end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time:.4f} s")
