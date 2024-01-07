import torch
import matplotlib.pyplot as plt
from torch import nn

weight = 0.7;
bias = 0.3;

start = 0;
end = 8;
step = 0.2;

x = torch.arange(start, end, step).unsqueeze(dim=1)
y= weight*x + bias

x[:40], y[:40]

# Create train/test split
train_split = int(0.8 * len(x)) # 80% of data used for training set, 20% for testing
X_train, y_train = x[:train_split], y[:train_split]
X_test, y_test = x[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)
print(y_test)

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
  
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14})
  plt.show();
  
plot_predictions();
  
  
class LinearRegg(nn.Module):
  def __init__(self):
    super(LinearRegg, self).__init__()

    self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

  def forward(self, x: torch.tensor) -> torch.tensor:
    return self.weights * x + self.bias
  
torch.manual_seed(42)
model = LinearRegg()
print(list(model.parameters()))
print(model.state_dict())

with torch.inference_mode():
  y_pred = model(X_test);
  
print(y_pred)

plot_predictions(predictions=y_pred)

loss_fn = nn.L1Loss();

optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

epochs = 100;

train_loss_values=[]
test_loss_values=[]
epoch_count=[]

for epoch in range(epochs):
  
  model.train()
  
  y_pred = model(X_train)
  
  loss = loss_fn(y_pred, y_train)
  
  optimizer.zero_grad()
  
  loss.backward()
  
  optimizer.step()
  
  model.eval()
  
  with torch.inference_mode():
    test_pred = model(X_test)
    
    test_loss = loss_fn(test_pred, y_test.type(torch.float))
    
    if epoch % 10 == 0:
      epoch_count.append(epoch)
      train_loss_values.append(loss.detach().numpy())
      test_loss_values.append(test_loss.detach().numpy())
      
      print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
      

plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend();
plt.show()

print(model.state_dict())

model.eval()

with torch.inference_mode():
  y_preds = model(X_test)
plot_predictions(predictions=y_preds)

from pathlib import Path

# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH) 