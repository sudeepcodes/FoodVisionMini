import torch
from torchvision.transforms import transforms

from data_setup import create_dataloaders
from engine import train
from model_builder import TinyVGG
from utils import save_model

# Setup Hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup Directories
train_dir = 'data/pizza_steak_sushi/train'
test_dir = 'data/pizza_steak_sushi/test'

# Setup target device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create Transform
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Create DataLoaders
train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                    test_dir=test_dir,
                                                                    transform=data_transform,
                                                                    batch_size=BATCH_SIZE)

# Create Model
model = TinyVGG(input_shape=3,
                hidden_units=HIDDEN_UNITS,
                output_shape=len(class_names)).to(device)

# Setup loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training the model
train(model=model,
      train_dataloader=train_dataloader,
      test_dataloader=test_dataloader,
      loss_fn=loss_fn,
      optimizer=optimizer,
      epochs=NUM_EPOCHS,
      device=device)

# Save the model with help from utils.py
save_model(model=model,
           target_dir="models",
           model_name="05_going_modular_script_mode_tinyvgg_model.pth")
