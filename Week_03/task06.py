import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def main():
    data = pd.read_csv('../DATA/ds_salaries.csv')

    features = ['experience_level', 'employment_type', 'remote_ratio', 'company_size']
    features_data = data[features]
    targets_data = data['salary_in_usd']

    # Encode categorical features
    encoders = {}
    for col in ['experience_level', 'employment_type', 'company_size']:
        encoders[col] = LabelEncoder()
        features_data[col] = encoders[col].fit_transform(features_data[col])

    # Normalize features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    features_data = scaler_X.fit_transform(features_data)
    targets_data = scaler_y.fit_transform(targets_data.values.reshape(-1, 1)).flatten()

    # Convert to tensor
    features_tensor = torch.tensor(features_data, dtype=torch.float32)
    targets_tensor = torch.tensor(targets_data, dtype=torch.float32)

    # Create Dataset and DataLoader
    dataset = TensorDataset(features_tensor, targets_tensor)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Define model architecture
    salary_predictor_model_sigmoid = nn.Sequential(
        nn.Linear(4, 16),
        nn.Sigmoid(),
        nn.Linear(16, 8),
        nn.Sigmoid(),
        nn.Linear(8, 1)
    )

    salary_predictor_model_relu = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )

    salary_predictor_model_lrelu = nn.Sequential(
        nn.Linear(4, 16),
        nn.LeakyReLU(),
        nn.Linear(16, 8),
        nn.LeakyReLU(),
        nn.Linear(8, 1)
    )

    # Training loop function
    def train_model(model, dataloader, epochs=20, lr=0.001):
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                predictions = model(batch_X).squeeze()
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
        return losses

    # Train different models
    results = {}

    test_models = {
        'sigmoid': salary_predictor_model_sigmoid,
        'relu': salary_predictor_model_relu,
        'leakyrely': salary_predictor_model_lrelu
    }

    for name, model in test_models.items():
        print(f"\nTraining model with {name} activation:\n")
        losses = train_model(model, dataloader)
        results[name] = losses

    plt.figure(figsize=(18, 6))

    for i, (name, losses) in enumerate(results.items(), 1):
        plt.subplot(1, 3, i)
        plt.plot(losses, label=f'{name} Activation', color='royalblue')
        plt.xlabel('Epochs')
        plt.ylabel('Average Training Loss')
        plt.title(f'nn_with_{name}')

    plt.suptitle('Training Loss Comparison for Different Activations', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Determine the best model
    final_losses = {name: losses[-1] for name, losses in results.items()}
    best_model = min(final_losses, key=final_losses.get)
    print(f"\nBest model: {best_model} with final loss: {final_losses[best_model]:.4f}")


if __name__ == '__main__':
    main()
