import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LogicMachine(nn.Module):
    def __init__(self):
        super(LogicMachine, self).__init__()
        
        # Layer definitions
        self.layer1 = nn.Linear(3, 16)  # 3 inputs (1 opcode + 2 values)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 3)   # 3 outputs
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Forward pass
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)              # No activation on final layer
        return x

def train_model(model, X_train, y_train, epochs=1000, learning_rate=0.001):
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        # Convert numpy arrays to torch tensors if needed
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.FloatTensor(X_train)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.FloatTensor(y_train)
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def prepare_data(opcodes, values1, values2, results):
    # Combine inputs
    X = np.column_stack((opcodes, values1, values2))
    y = np.array(results)
    return X, y

# Example usage:
if __name__ == "__main__":
    # Create the model
    model = LogicMachine()
    
    # Example data (you would replace this with your actual dataset)
    # Let's say opcodes are: 0=add, 1=subtract, 2=multiply, 3=divide
    X_train = np.array([
        [0, 2, 3],    # add 2 + 3
        [1, 5, 2],    # subtract 5 - 2
        [2, 4, 3],    # multiply 4 * 3
        [3, 8, 2],    # divide 8 / 2
    ], dtype=np.float32)
    
    y_train = np.array([
        [5, 0, 0],    # result of 2 + 3
        [3, 0, 0],    # result of 5 - 2
        [12, 0, 0],   # result of 4 * 3
        [4, 0, 0],    # result of 8 / 2
    ], dtype=np.float32)
    
    # Train the model
    train_model(model, X_train, y_train)
    
    # Test the model
    test_input = torch.FloatTensor([[0, 6, 4]])  # Testing addition of 6 + 4
    prediction = model(test_input)
    print(f"\nTest prediction for 6 + 4: {prediction.detach().numpy()}")
