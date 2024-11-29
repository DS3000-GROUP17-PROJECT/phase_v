import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load your data into pandas DataFrame (adjust this to your dataset)
df = pd.read_csv('american_bankruptcy.csv')


# 2. Preprocess the data

# Encode 'status_label' (alive -> 1, failed -> 0)
df['status_label'] = df['status_label'].map({'alive': 1, 'failed': 0})

# Sort by company_name and year (for time-series modeling)
df = df.sort_values(by=['company_name', 'year'])

# Normalize the features (18 non-temporal features)
features_columns = [f'feature_{i}' for i in range(1, 19)]  # Assuming 18 features
scaler = StandardScaler()
df[features_columns] = scaler.fit_transform(df[features_columns])

# Create temporal windows for each company: Let's assume a window size of 3 years
window_size = 3

# Function to create temporal windows for time series data
def create_temporal_windows(df, window_size):
    windowed_data = []
    for _, group in df.groupby('company_name'):
        for i in range(len(group) - window_size + 1):
            windowed_data.append(group.iloc[i:i + window_size])
    return pd.DataFrame(windowed_data)

df_windowed = create_temporal_windows(df, window_size)

# Split data into X (features) and y (target)
X = df_windowed[features_columns].values
y = df_windowed['status_label'].values

# Train-test split (stratify to preserve the class balance in both train and test sets)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Compute class weights for imbalanced classes
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 3. Define CNN model for feature extraction
class CNN_Model(nn.Module):
    def __init__(self, input_size):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(64 * (input_size - 4), 128)  # Adjust size based on window size and kernel
        self.fc2 = nn.Linear(128, 2)  # Output layer for binary classification (alive vs failed)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, channels, features) for 1D CNN
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Output raw scores (no softmax)
        return x

# 4. Initialize CNN model, loss function, and optimizer
model = CNN_Model(input_size=X_train_tensor.shape[1])

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)  # Account for class imbalance
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Train the CNN model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 6. Extract features using the trained CNN model
with torch.no_grad():
    model.eval()
    cnn_features_train = model.fc1(model.conv2(model.conv1(X_train_tensor).relu()).relu()).numpy()
    cnn_features_test = model.fc1(model.conv2(model.conv1(X_test_tensor).relu()).relu()).numpy()

# 7. Train Logistic Regression on CNN features
log_reg_model = LogisticRegression(class_weight='balanced', max_iter=1000)
log_reg_model.fit(cnn_features_train, y_train)

# 8. Make predictions on the test set
y_pred = log_reg_model.predict(cnn_features_test)

# 9. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
