import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader
import obspy
from obspy import read
from obspy import UTCDateTime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

try:
    st = read('../Data/0121Dapu.mseed')
except Exception as e:
    print(f"Error loading waveform data: {e}")
    st = None

if st:
    t1 = UTCDateTime(2025, 1, 20, 16, 17, 16)
    t2 = UTCDateTime(2025, 1, 20, 16, 17, 34)

    st_sliced_1 = st.slice(t1, t1 + 5)
    st_sliced_2 = st.slice(t2, t2 + 5) # Use t2 to t2+5 as per your example

    st2 = st_sliced_2.copy() # Work on a copy to avoid modifying the original stream

    # Perform normalization and detrending
    st2.normalize()
    st2.detrend()

    processed_data = []
    labels = []

    all_processed_traces = [i.data for i in st2] # Get data from all traces in st2

    # First 129 traces in st2 are for training
    X_train_data = all_processed_traces[:129]
    Y_train_labels = [1 if st2[i].stats.station in ["ALS", "CHN8", "CHY", "SCL", "ELD"] else 0 for i in range(129)]

    # Remaining traces in st2 are for testing
    X_test_data = all_processed_traces[129:]
    Y_test_labels = [1 if st2[i+129].stats.station in ["SSD", "TAI", "TAI1", "WDL", "WGK", "WSF", "YUS"] else 0 for i in range(len(st2) - 129)]

    print("Waveform data loaded and processed with labels.")
    print(f"Number of training data points: {len(X_train_data)}")
    print(f"Number of testing data points: {len(X_test_data)}", "\n")

else:
    print("Failed to load waveform data.")

class EarthquakeCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super(EarthquakeCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        self.fc1 = None
        self.fc2 = nn.Linear(64, num_classes) # num_classes is 2 for earthquake/non-earthquake

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        if self.fc1 is None:
            flattened_size = x.view(x.size(0), -1).size(1)
            self.fc1 = nn.Linear(flattened_size, 64).to(x.device)

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
model = EarthquakeCNN(input_channels=1, num_classes=2)

print(model,"\n")

# Convert the training data to PyTorch tensors
X_train_tensor = torch.tensor(np.array(X_train_data), dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_labels, dtype=torch.long)

# Add a channel dimension to the input tensor
X_train_tensor = X_train_tensor.unsqueeze(1) # Shape becomes (samples, 1, sequence_length)

# Create a TensorDataset from the tensors
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)

# Create a DataLoader from the TensorDataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Re-instantiate the model with the correct input length from the tensor
# input_length = X_train_tensor.size(2)  # Get the length from your existing tensor
# flattened_size = 64 * (input_length // 4)
# model = EarthquakeCNN(input_channels=1, num_classes=2)

print("Training data prepared for DataLoader.")
print(f"Shape of X_train_tensor: {X_train_tensor.shape}")
print(f"Shape of Y_train_tensor: {Y_train_tensor.shape}", "\n")

import torch.optim as optim
from torch.nn import CrossEntropyLoss

model = EarthquakeCNN(input_channels=1, num_classes=2)

# Define the loss function and optimizer
criterion = CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Define the number of epochs for training
num_epochs = 500

# Move the model to the appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
print("Starting training...")
for epoch in range(num_epochs):
    model.train() # Set the model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Ensure inputs have an extra dimension for channels (batch_size, channels, sequence_length)
        # Current shape is likely (batch_size, sequence_length)
        # inputs = inputs.unsqueeze(1) # Remove redundant unsqueeze(1)

        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0) # Accumulate loss

    epoch_loss = running_loss / len(train_dataset)
    if (epoch + 1) % 50 == 0:
      print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

print("Training finished.", "\n")

model.eval()

X_test_tensor = torch.tensor(np.array(X_test_data), dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test_labels, dtype=torch.long)
X_test_tensor = X_test_tensor.unsqueeze(1)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

classified_as_00 = []
classified_as_11 = []
misclassified_as_01 = [] # True label is 1, but predicted as 0
misclassified_as_10 = [] # True label is 0, but predicted as 1
all_labels = []
all_predictions = []

with torch.no_grad():
    inputs = X_test_tensor.to(device)
    labels = Y_test_tensor.to(device)

    outputs = model(inputs)

    _, predicted = torch.max(outputs.data, 1)
    
    # Use to confusion matrix
    all_labels = labels.cpu().numpy().tolist()
    all_predictions = predicted.cpu().numpy().tolist()

    # Compare predictions with true labels
    for i in range(len(labels)):
        true_label = labels[i].item()
        predicted_label = predicted[i].item()
        
        # Get station and channel info from the original obspy stream
        original_trace_index = i + 129
        station_name = st_sliced_2[original_trace_index].stats.station
        channel_name = st_sliced_2[original_trace_index].stats.channel

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


# Create a single figure with 4 rows and 3 columns for subplots
fig, axes = plt.subplots(4, 3, figsize=(15, 15))
fig.suptitle("Testing Examples - Model Performance Overview", fontsize=16)

# Define the list of classified examples for each category and their color
categories = {
    "Label: 0, Predict: 0": {"examples": classified_as_00, "color": "#1f77b4"}, # Blue for correctly classified
    "Label: 1, Predict: 1": {"examples": classified_as_11, "color": "#1f77b4"}, # Blue for correctly classified
    "Label: 0, Predict: 1 (Misclassified)": {"examples": misclassified_as_10, "color": "#d62728"}, # Red for misclassified
    "Label: 1, Predict: 0 (Misclassified)": {"examples": misclassified_as_01, "color": '#d62728'}  # Red for misclassified
}

# Iterate through categories and plot examples
for i, (title, category_info) in enumerate(categories.items()):
    examples = category_info["examples"]
    color = category_info["color"]
    # Randomly sample up to 3 examples
    sampled_examples = random.sample(examples, min(len(examples), 3))

    # Plot each sampled example in a separate subplot in the current row
    for j in range(len(sampled_examples)): # Iterate only up to the number of sampled examples
        ax = axes[i, j]
        data, label, predict, station_name, channel_name = sampled_examples[j]
        new_title = f"Station: {station_name}, Channel: {channel_name}\n{title}"
        ax.plot(data.numpy()[0], color=color) # Use the specified color
        ax.set_title(new_title, fontsize=10)
        ax.set_xticks(np.linspace(0, len(data[0]) - 1, 6))
        ax.set_xticklabels(np.linspace(0, 5, 6))
        ax.set_xlabel("Time (s)")

    # Hide remaining subplots in the row if fewer than 3 examples were plotted
    for j in range(len(sampled_examples), 3):
        fig.delaxes(axes[i, j])


plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.savefig("test_examples_overview.png", dpi=300)
print("[PNG]test_examples_overview be stored", "\n")


if not all_labels or not all_predictions:
    print("Warning: The label and prediction lists are empty. Please run the model evaluation first.")
else:
    # --- Confusion Matrix and Metrics ---
    # Convert lists to numpy arrays for calculation
    all_labels_np = np.array(all_labels)
    all_predictions_np = np.array(all_predictions)
    total = len(all_labels_np)

    # Calculate the confusion matrix
    cm = confusion_matrix(all_labels_np, all_predictions_np)
    
    # Unpack the confusion matrix
    tn, fp, fn, tp = cm.ravel()

    # Calculate rates
    correct_rate = (tp + tn) / total
    incorrect_rate = (fp + fn) / total

    print("\n--- Confusion matrix ---")
    print(f"Total: {total}")
    print(f"Correct: {tp + tn} ({correct_rate:.2%})")
    print(f"Miscorrect: {fp + fn} ({incorrect_rate:.2%})")
    print("-" * 30)
    print(f"True Positives (TP): {tp} (label 1, Pred 0)")
    print(f"True Negatives (TN): {tn} (label 0, Pred 0)")
    print(f"False Positives (FP): {fp} (label 0, Pred 1 - Mis)")
    print(f"False Negatives (FN): {fn} (label 1, Pred 0 - Omit)")

    # Plotting the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-earthquake (0)", "Earthquake (1)"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    print("\n[PNG] confusion_matrix ve stored")
    plt.close(fig)
