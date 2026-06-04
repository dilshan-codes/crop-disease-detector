import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model, test_loader, full_dataset, device, checkpoint_path='models/best_model.pth'):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(
        all_labels, all_preds,
        target_names=full_dataset.classes
    ))

    test_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Greens',
        xticklabels=[c[:20] for c in full_dataset.classes],
        yticklabels=[c[:20] for c in full_dataset.classes]
    )
    plt.title('Confusion Matrix — Test Set', fontsize=14)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()