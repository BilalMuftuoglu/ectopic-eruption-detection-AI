from transformers import AutoModelForImageClassification, AutoImageProcessor
from datasets import load_dataset
import torch
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import subprocess

def print_confusion_matrix(cm,tooth,accuracy):
  class_names = ["Mild", "Moderate","Severe"]  # Replace with your actual class names
  sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
  plt.xlabel("Predicted")
  plt.ylabel("True")
  plt.title(f"Confusion Matrix for Diseased Tooth {tooth} (deit model)\nAccuracy: {accuracy}")
  plt.savefig(f"confusion_matrix_tooth_{tooth}_diseased_deit.png")  # Save the figure
  plt.show()

def pad_matrix(matrix, target_shape):
    """Pads the given matrix to the target shape with zeros."""
    result = np.zeros(target_shape, dtype=int)
    result[:matrix.shape[0], :matrix.shape[1]] = matrix
    return result

model_checkpoint = "facebook/deit-base-distilled-patch16-224"
#model_checkpoint = "microsoft/beit-base-patch16-224" 

tooths = ["55","65","75","85"]
all_accuracies = []
all_names = []
for tooth in tooths:
  cm = np.matrix(np.zeros((3,3),dtype=int))
  gtruth_labels = []
  pred_labels = []
  accuracies = []
  for f in range(5):
    model_name = model_checkpoint.split("/")[-1]
    repo_name = f"{model_name}-hasta-{tooth}-fold{f+1}"
    #how to fetch a repo to local
    #!git clone https://huggingface.co/BilalMuftuoglu/{repo_name}
    git_clone_command = f"https://huggingface.co/BilalMuftuoglu/{repo_name}"
    subprocess.run(['git', 'clone', git_clone_command])

    image_processor = AutoImageProcessor.from_pretrained(repo_name)
    model = AutoModelForImageClassification.from_pretrained(repo_name)

    test_dataset = load_dataset(f"dataset/k-fold-all/{tooth}/fold{f+1}/test", data_dir="")

    ground_truth_labels = test_dataset["train"]["label"]
    predicted_labels = []
    for image in test_dataset["train"]["image"]:
        encoding = image_processor(image.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
          outputs = model(**encoding)
          logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_labels.append(predicted_class_idx)

    conf_matrix = confusion_matrix(ground_truth_labels, predicted_labels)

    if conf_matrix.shape != cm.shape:
      conf_matrix = pad_matrix(conf_matrix, cm.shape)

    a = np.matrix(conf_matrix)
    cm = cm + a

    gtruth_labels.extend(ground_truth_labels)
    pred_labels.extend(predicted_labels)
    #how to get accuracy from model
    accuracy = (conf_matrix[0][0] + conf_matrix[1][1]) / np.sum(conf_matrix)
    accuracies.append(accuracy)
    all_accuracies.append(accuracy)
    all_names.append(f"{tooth} Fold {f+1}")
    print(accuracy)

  #calculate_precision_recall_f1(gtruth_labels, pred_labels)
  print_confusion_matrix(cm,tooth,np.sum(accuracies) / len(accuracies))

plt.figure(figsize=(10, 6))
colors = ['skyblue'] *5 +['lightgreen']*5 +  ['lightcoral']*5 + ['orange']*5
bars = plt.bar(all_names, all_accuracies, color=colors)

# Değerleri kutuların yanında gösterme
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval , round(yval,2), ha='center', va='bottom')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.title('Accuracy values ​​of Ectopic Eruption disease classification (deit model)')
plt.savefig('accuracy_values_diseased_deit.png')
plt.show()

##############################################################################################################

def print_confusion_matrix_diseased(cm,tooth,accuracy):
  class_names = ["Diseased", "Normal"]  # Replace with your actual class names
  sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
  plt.xlabel("Predicted")
  plt.ylabel("True")
  plt.title(f"Confusion Matrix for Tooth {tooth} (beit model)\nAccuracy: {accuracy}")
  plt.savefig(f"confusion_matrix_tooth_{tooth}_beit.png")  # Save the figure
  plt.show()


def calculate_precision_recall_f1_diseased(gtruth_labels, pred_labels):
  report = classification_report(gtruth_labels, pred_labels, target_names=["Diseased", "Normal"], output_dict=True)

  # Precision, recall ve f1-score değerlerini alma
  precision = [report['Diseased']['precision'], report['Normal']['precision']]
  recall = [report['Diseased']['recall'], report['Normal']['recall']]
  f1_score = [report['Diseased']['f1-score'], report['Normal']['f1-score']]
  labels = ['Diseased', 'Normal']

  # Grafiği oluşturma
  x = np.arange(len(labels))  # the label locations
  width = 0.15  # the width of the bars

  fig, ax = plt.subplots()
  rects1 = ax.bar(x - width, precision, width, label='Precision')
  rects2 = ax.bar(x, recall, width, label='Recall')
  rects3 = ax.bar(x + width, f1_score, width, label='F1-Score')

  # Eksen etiketlerini ayarlama
  ax.set_xlabel('Classes')
  ax.set_ylabel('Scores')
  ax.set_title(f'Precision, Recall, and F1-Score for Tooth {tooth} (beit model)')
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  ax.legend()

  # Çubukların üzerine değerleri ekleme
  def autolabel(rects):
      """Çubukların üzerine değerleri ekleme fonksiyonu"""
      for rect in rects:
          height = rect.get_height()
          ax.annotate('%.2f' % height,
                      xy=(rect.get_x() + rect.get_width() / 2, height),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha='center', va='bottom')

  autolabel(rects1)
  autolabel(rects2)
  autolabel(rects3)

  fig.tight_layout()

  # Grafiği gösterme
  plt.savefig(f'precision_recall_f1_tooth_{tooth}.png')
  plt.show()



tooths = ["55","65","75","85"]
all_accuracies = []
all_names = []
for tooth in tooths:
  cm = np.matrix(np.zeros((2,2),dtype=int))
  gtruth_labels = []
  pred_labels = []
  accuracies = []
  for f in range(5):
    model_name = model_checkpoint.split("/")[-1]
    repo_name = f"{model_name}-{tooth}-fold{f+1}"
    #how to fetch a repo to local
    #!git clone https://huggingface.co/BilalMuftuoglu/{repo_name}
    git_clone_command = f"https://huggingface.co/BilalMuftuoglu/{repo_name}"
    subprocess.run(['git', 'clone', git_clone_command])

    image_processor = AutoImageProcessor.from_pretrained(repo_name)
    model = AutoModelForImageClassification.from_pretrained(repo_name)

    test_dataset = load_dataset(f"dataset/k-fold-all/{tooth}/fold{f+1}/test", data_dir="")

    ground_truth_labels = test_dataset["train"]["label"]
    predicted_labels = []
    for image in test_dataset["train"]["image"]:
        encoding = image_processor(image.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
          outputs = model(**encoding)
          logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_labels.append(predicted_class_idx)

    conf_matrix = confusion_matrix(ground_truth_labels, predicted_labels)
    a = np.matrix(conf_matrix)
    cm = cm + a
    print(gtruth_labels)
    print(pred_labels)
    print(conf_matrix)
    print(cm)
    gtruth_labels.extend(ground_truth_labels)
    pred_labels.extend(predicted_labels)
    #how to get accuracy from model
    accuracy = (conf_matrix[0][0] + conf_matrix[1][1]) / np.sum(conf_matrix)
    accuracies.append(accuracy)
    all_accuracies.append(accuracy)
    all_names.append(f"{tooth} Fold {f+1}")
    print(accuracy)
    #!rm -rf {model_name}-{tooth}-fold{f+1}

  #calculate_precision_recall_f1_diseased(gtruth_labels, pred_labels)
  print_confusion_matrix_diseased(cm,tooth,np.sum(accuracies) / len(accuracies))


plt.figure(figsize=(10, 6))
colors = ['skyblue'] *5 +['lightgreen']*5 +  ['lightcoral']*5 + ['orange']*5
bars = plt.bar(all_names, all_accuracies, color=colors)

# Değerleri kutuların yanında gösterme
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval , round(yval,2), ha='center', va='bottom')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.title('Accuracy values ​​of Ectopic Eruption binary classification (beit model)')
plt.savefig('accuracy_values.png')
plt.show()





