import matplotlib.pyplot as plt


def plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies, output_dir="output"):
    # Plot Loss
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss Across Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    loss_plot_path = f"{output_dir}/loss_plot.png"
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.title("Accuracy Across Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    accuracy_plot_path = f"{output_dir}/accuracy_plot.png"
    plt.savefig(accuracy_plot_path)
    print(f"Accuracy plot saved to {accuracy_plot_path}")
    plt.close()
