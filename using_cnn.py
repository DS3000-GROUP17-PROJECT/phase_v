import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt

# Load your dataframe
df = pd.read_csv("american_bankruptcy.csv")

# Preprocessing
# Normalize X1, X2, ..., X18
scaler = StandardScaler()
features = df.iloc[:, 3:].values  # X1 to X18
features_scaled = scaler.fit_transform(features)

# Convert categorical columns into numerical representations
df['company_name'] = pd.factorize(df['company_name'])[0]
df['status_label'] = pd.factorize(df['status_label'])[0]
df['year'] = df['year'].astype('category').cat.codes

# Prepare input data and target labels
X = df[['company_name', 'status_label', 'year']].values  # Use company_name, status_label, and year as additional features
X = torch.tensor(features_scaled, dtype=torch.float32)  # Feature columns X1 to X18
y = torch.tensor(df['status_label'].values, dtype=torch.long)  # Assuming you want to predict 'status_label'

# Reshaping the input data to fit CNN (batch_size, channels, sequence_length)
X = X.unsqueeze(1)  # Add a channel dimension for the 1D CNN

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define CNN model
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # 1D Convolution layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(16 * 9, 64)  # Assuming the length of the sequence is 18 (X1 to X18), pooling reduces it
        self.fc2 = nn.Linear(64, 2)  # 2 output classes (alive, dead)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 9)  # Flatten for the fully connected layer
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# Initialize model, loss, and optimizer
model = CNN_Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
"""
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    _, predicted = torch.max(y_pred, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
"""
# Extract features from CNN model (not using final layer for classification)
def extract_features(model, X_data):
    model.eval()
    with torch.no_grad():
        features = model.conv1(X_data)
        features = torch.relu(features)
        features = model.pool(features)
        features = features.view(features.size(0), -1)  # Flatten for the fully connected layer
        return features
    
# Train the CNN model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train)
    loss = nn.CrossEntropyLoss()(outputs, y_train)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Extract CNN features for logistic regression
X_train_features = extract_features(model, X_train).numpy()
X_test_features = extract_features(model, X_test).numpy()

# Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, class_weight= 'balanced', penalty= 'l2')

# Train the Logistic Regression model
log_reg.fit(X_train_features, y_train)

# Make predictions
y_pred = log_reg.predict(X_test_features)
y_pred_prod = log_reg.predict_proba(X_test_features)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 4-fold Cross-validation
cv_scores = cross_val_score(log_reg, X_train_features, y_train, cv=4)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {cv_scores.mean():.2f}")

def plot_prc(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    prc_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', label=f'PRC (AUC = {prc_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Call the plot_prc function
plot_prc(y_test, y_pred_prod)