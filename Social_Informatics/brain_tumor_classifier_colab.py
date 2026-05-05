# ============================================================
#         Brain Tumor Classification - Google Colab Version
# ============================================================
# Models: Random Forest, Logistic Regression, SVC,
#         Decision Tree, KNN, Soft Voting Ensemble
# Metrics: Accuracy, Precision, Recall, Specificity,
#          F1-Score, AUC, ROC Curve, Confusion Matrix,
#          Model Comparison Chart, Overfitting Check
# ============================================================

# ====================== Install Dependencies ======================
# Uncomment and run this cell first in Colab:
# !pip install numpy opencv-python-headless scikit-learn seaborn matplotlib Pillow joblib

# ====================== Imports ======================
import os
import numpy as np
import cv2
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, StandardScaler

# Models
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc
)

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import display
from PIL import Image

# Google Colab file upload
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ====================== Constants ======================
CLASS_NAMES   = ['Normal', 'Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor']
CLASS_MAPPING = {'normal': 0, 'glioma_tumor': 1, 'meningioma_tumor': 2, 'pituitary_tumor': 3}
IMG_SIZE      = 128
DATA_DIR      = "brain_tumor_dataset"
MODEL_PATH    = "brain_tumor_best_model.pkl"
N_CLASSES     = len(CLASS_NAMES)


# ============================================================
#                   1. DATA LOADING
# ============================================================
def load_data(data_dir=DATA_DIR, img_size=IMG_SIZE):
    """Load MRI images from subfolders, resize, flatten, and return X and y."""
    images, labels = [], []

    print("📂 Loading dataset...")
    for class_name, class_label in CLASS_MAPPING.items():
        class_dir = os.path.join(data_dir, class_name)

        if not os.path.exists(class_dir):
            print(f"  ⚠️  Folder not found: {class_dir} — skipping.")
            continue

        count = 0
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                images.append(img.flatten())
                labels.append(class_label)
                count += 1

        print(f"  ✅ {class_name}: {count} images loaded.")

    X = np.array(images)
    y = np.array(labels)
    print(f"\n📊 Total dataset — X: {X.shape}, y: {y.shape}\n")
    return X, y


# ============================================================
#                   2. PREPROCESSING
# ============================================================
def preprocess(X_train, X_test):
    """Normalize pixel values using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ============================================================
#                   3. MODEL DEFINITIONS
# ============================================================
def get_models():
    """Return a dictionary of all classifiers to train and evaluate."""
    # Define base models
    rf  = RandomForestClassifier(n_estimators=100, random_state=42)
    lr  = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
    svc = SVC(kernel='rbf', probability=True, random_state=42)
    dt  = DecisionTreeClassifier(random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)

    # Soft Voting Ensemble — combines all 5 models
    # Uses average of predicted probabilities (soft voting)
    ensemble = VotingClassifier(
        estimators=[
            ('rf',  rf),
            ('lr',  lr),
            ('svc', svc),
            ('dt',  dt),
            ('knn', knn),
        ],
        voting='soft'   # averages probabilities instead of majority vote
    )

    return {
        "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr'),
        "SVC"                 : SVC(kernel='rbf', probability=True, random_state=42),
        "Decision Tree"       : DecisionTreeClassifier(random_state=42),
        "KNN"                 : KNeighborsClassifier(n_neighbors=5),
        "Ensemble (Soft Vote)": ensemble,
    }


# ============================================================
#                   4. METRICS CALCULATION
# ============================================================
def compute_specificity(y_test, y_pred, n_classes=N_CLASSES):
    """
    Compute per-class specificity (true negative rate) and return the macro average.
    Specificity = TN / (TN + FP)
    """
    y_bin     = label_binarize(y_test, classes=list(range(n_classes)))
    y_pred_bin = label_binarize(y_pred, classes=list(range(n_classes)))

    specificities = []
    for i in range(n_classes):
        tn = np.sum((y_bin[:, i] == 0) & (y_pred_bin[:, i] == 0))
        fp = np.sum((y_bin[:, i] == 0) & (y_pred_bin[:, i] == 1))
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)

    return np.mean(specificities)


def compute_metrics(model, X_train, y_train, X_test, y_test):
    """
    Train model and compute all evaluation metrics.
    Returns a dict of metric values and prediction arrays.
    """
    # ── Train ──────────────────────────────────────────────
    model.fit(X_train, y_train)

    # ── Predictions ────────────────────────────────────────
    y_pred       = model.predict(X_test)
    y_proba      = model.predict_proba(X_test)        # shape (n_samples, n_classes)
    train_acc    = accuracy_score(y_train, model.predict(X_train))

    # ── Core metrics ───────────────────────────────────────
    test_acc     = accuracy_score(y_test, y_pred)
    precision    = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall       = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1           = f1_score(y_test, y_pred, average='macro', zero_division=0)
    specificity  = compute_specificity(y_test, y_pred)

    # ── AUC (One-vs-Rest, macro) ───────────────────────────
    y_test_bin   = label_binarize(y_test, classes=list(range(N_CLASSES)))
    auc_score    = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro')

    return {
        "train_acc"  : train_acc,
        "test_acc"   : test_acc,
        "precision"  : precision,
        "recall"     : recall,
        "f1"         : f1,
        "specificity": specificity,
        "auc"        : auc_score,
        "y_pred"     : y_pred,
        "y_proba"    : y_proba,
    }


# ============================================================
#                   5. CONFUSION MATRIX PLOT
# ============================================================
def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plot a styled confusion matrix for a single model."""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, linecolor='gray'
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{model_name.replace(' ', '_')}.png", dpi=150)
    plt.show()


# ============================================================
#                   6. ROC CURVE PLOT
# ============================================================
def plot_roc_curve(y_test, y_proba, model_name):
    """Plot ROC curve (one curve per class) for a single model."""
    y_test_bin = label_binarize(y_test, classes=list(range(N_CLASSES)))
    colors     = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    plt.figure(figsize=(8, 6))
    for i, (class_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc     = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f"{class_name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curve — {model_name}", fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"roc_curve_{model_name.replace(' ', '_')}.png", dpi=150)
    plt.show()


# ============================================================
#             7. MODEL COMPARISON CHART
# ============================================================
def plot_model_comparison(results_dict):
    """Bar chart comparing all models across key metrics."""
    model_names = list(results_dict.keys())
    metrics     = ["test_acc", "precision", "recall", "f1", "specificity", "auc"]
    labels      = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "AUC"]
    colors      = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']

    x     = np.arange(len(model_names))
    width = 0.13

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        values = [results_dict[m][metric] for m in model_names]
        bars   = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.2f}",
                ha='center', va='bottom', fontsize=7, rotation=45
            )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — All Metrics", fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.show()


# ============================================================
#             8. OVERFITTING CHECK CHART
# ============================================================
def plot_overfitting_check(results_dict):
    """Bar chart comparing training vs testing accuracy for each model."""
    model_names = list(results_dict.keys())
    train_accs  = [results_dict[m]["train_acc"] for m in model_names]
    test_accs   = [results_dict[m]["test_acc"]  for m in model_names]

    x     = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, train_accs, width, label='Training Accuracy',
                   color='#3498db', alpha=0.85)
    bars2 = ax.bar(x + width/2, test_accs,  width, label='Testing Accuracy',
                   color='#e74c3c', alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=9)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Training vs Testing Accuracy (Overfitting Check)",
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("overfitting_check.png", dpi=150)
    plt.show()


# ============================================================
#             9. RESULTS SUMMARY TABLE
# ============================================================
def print_summary_table(results_dict):
    """Print a formatted summary table of all models and metrics."""
    header = f"\n{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Spec':>6} {'AUC':>6}"
    sep    = "-" * len(header)

    print("\n" + "=" * len(header))
    print("         📋  MODEL PERFORMANCE SUMMARY")
    print("=" * len(header))
    print(header)
    print(sep)

    best_model_name = max(results_dict, key=lambda m: results_dict[m]["auc"])

    for name, r in results_dict.items():
        marker = " ⭐" if name == best_model_name else ""
        print(
            f"{name:<22} "
            f"{r['test_acc']:>6.3f} "
            f"{r['precision']:>6.3f} "
            f"{r['recall']:>6.3f} "
            f"{r['f1']:>6.3f} "
            f"{r['specificity']:>6.3f} "
            f"{r['auc']:>6.3f}"
            f"{marker}"
        )

    print(sep)
    print(f"\n🏆 Best Model (by AUC): {best_model_name}")
    print("=" * len(header) + "\n")

    return best_model_name


# ============================================================
#             10. SAVE ENSEMBLE MODEL
# ============================================================
def save_best_model(models_dict, results_dict, scaler, best_model_name):
    """
    Always save the Ensemble model as the final model.
    Also prints comparison to show Ensemble vs individual models.
    """
    # ── Save Ensemble model ────────────────────────────────
    ensemble_model = models_dict["Ensemble (Soft Vote)"]
    joblib.dump({"model": ensemble_model, "scaler": scaler}, MODEL_PATH)
    print(f"✅ Ensemble (Soft Vote) model saved to '{MODEL_PATH}'")

    # ── Show why Ensemble is better ───────────────────────
    print("\n  📊 Ensemble vs Individual Models:")
    print(f"  {'Model':<25} {'AUC':>6}")
    print("  " + "-" * 33)
    for name, r in results_dict.items():
        marker = " ⭐" if name == "Ensemble (Soft Vote)" else ""
        print(f"  {name:<25} {r['auc']:>6.3f}{marker}")
    print(f"\n  🏆 Final model for prediction: Ensemble (Soft Vote)")


# ============================================================
#             11. PREDICTION ON NEW IMAGE (Colab)
# ============================================================
def predict_image(model_path=MODEL_PATH, img_size=IMG_SIZE):
    """
    Upload an MRI image in Colab and predict the tumor class.
    Falls back to a manual path prompt if not running in Colab.
    """
    # Load saved model and scaler
    saved      = joblib.load(model_path)
    model      = saved["model"]
    scaler     = saved["scaler"]

    # Get image from user
    if IN_COLAB:
        print("📤 Please upload an MRI image:")
        uploaded  = files.upload()
        img_path  = list(uploaded.keys())[0]
    else:
        img_path  = input("Enter full path to MRI image: ").strip()

    # Read and preprocess image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Error: Could not read image.")
        return

    img_resized = cv2.resize(img, (img_size, img_size))
    img_flat    = img_resized.flatten().reshape(1, -1)
    img_scaled  = scaler.transform(img_flat)

    # Predict
    prediction  = model.predict(img_scaled)[0]
    proba       = model.predict_proba(img_scaled)[0]
    confidence  = proba[prediction] * 100

    # ── Display image ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: MRI image
    axes[0].imshow(img_resized, cmap='gray')
    axes[0].set_title("Uploaded MRI Scan", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Right: Probability bar chart
    colors_bar = ['#2ecc71' if i == prediction else '#3498db' for i in range(N_CLASSES)]
    bars       = axes[1].barh(CLASS_NAMES, proba * 100, color=colors_bar, alpha=0.85)
    axes[1].set_xlabel("Probability (%)", fontsize=11)
    axes[1].set_title("Class Probabilities", fontsize=12, fontweight='bold')
    axes[1].set_xlim(0, 110)
    axes[1].grid(axis='x', alpha=0.3)

    for bar, prob in zip(bars, proba):
        axes[1].text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.1f}%", va='center', fontsize=10
        )

    result_color = '#27ae60' if prediction == 0 else '#e74c3c'
    fig.suptitle(
        f"Prediction: {CLASS_NAMES[prediction]}  |  Confidence: {confidence:.1f}%",
        fontsize=14, fontweight='bold', color=result_color, y=1.02
    )

    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=150, bbox_inches='tight')
    plt.show()

    # ── Print result ───────────────────────────────────────
    print("\n" + "=" * 45)
    print(f"  🧠 Prediction  : {CLASS_NAMES[prediction]}")
    print(f"  📈 Confidence  : {confidence:.1f}%")
    print("\n  📊 All Probabilities:")
    for cls, prob in zip(CLASS_NAMES, proba):
        bar_fill = "█" * int(prob * 30)
        print(f"     {cls:<22} {bar_fill:<30} {prob * 100:.1f}%")
    print("=" * 45 + "\n")


# ============================================================
#                   12. MAIN PIPELINE
# ============================================================
def main():
    print("=" * 55)
    print("   🧠  Brain Tumor Classification — Full Pipeline")
    print("   Models: 5 Classifiers + Soft Voting Ensemble")
    print("=" * 55)

    # ── 1. Load data ───────────────────────────────────────
    X, y = load_data()

    # ── 2. Train/test split ────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"📌 Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}\n")

    # ── 3. Preprocess (scale) ──────────────────────────────
    X_train_s, X_test_s, scaler = preprocess(X_train, X_test)

    # ── 4. Train all models & collect metrics ──────────────
    models_dict  = get_models()
    results_dict = {}

    for model_name, model in models_dict.items():
        print(f"⚙️  Training: {model_name} ...")
        results_dict[model_name] = compute_metrics(
            model, X_train_s, y_train, X_test_s, y_test
        )
        print(f"   ✅ Done — Test Accuracy: {results_dict[model_name]['test_acc']:.3f} | "
              f"AUC: {results_dict[model_name]['auc']:.3f}")

    # ── 5. Detailed report per model ───────────────────────
    print("\n\n📋 Detailed Classification Reports:")
    print("=" * 55)
    for model_name, r in results_dict.items():
        print(f"\n🔹 {model_name}")
        print(classification_report(y_test, r["y_pred"], target_names=CLASS_NAMES))

    # ── 6. Confusion matrices ──────────────────────────────
    print("\n📊 Plotting Confusion Matrices...")
    for model_name, r in results_dict.items():
        plot_confusion_matrix(y_test, r["y_pred"], model_name)

    # ── 7. ROC curves ──────────────────────────────────────
    print("\n📈 Plotting ROC Curves...")
    for model_name, r in results_dict.items():
        plot_roc_curve(y_test, r["y_proba"], model_name)

    # ── 8. Model comparison chart ──────────────────────────
    print("\n📊 Plotting Model Comparison...")
    plot_model_comparison(results_dict)

    # ── 9. Overfitting check ───────────────────────────────
    print("\n📊 Plotting Overfitting Check...")
    plot_overfitting_check(results_dict)

    # ── 10. Summary table ──────────────────────────────────
    best_model_name = print_summary_table(results_dict)

    # ── 11. Save Ensemble model ────────────────────────────
    save_best_model(models_dict, results_dict, scaler, best_model_name)

    # ── 12. Auto-download the saved Ensemble model ─────────
    if IN_COLAB:
        print("\n💾 Downloading Ensemble model to your computer...")
        from google.colab import files
        files.download(MODEL_PATH)
        print("✅ Model downloaded — keep it safe for future predictions!")

    # ── 13. Predict on new image ───────────────────────────
    print("\n🔍 Ready to predict on a new MRI image!")
    predict_image()


# ── Entry point ────────────────────────────────────────────
if __name__ == "__main__":
    main()
