import re
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

lstm_64_250_1 = [
"accuracy: 0.7659 - loss: 0.4763 - val_accuracy: 0.7558 - val_loss: 0.5339",
"accuracy: 0.9000 - loss: 0.2596 - val_accuracy: 0.8041 - val_loss: 0.4749",
"accuracy: 0.9160 - loss: 0.2173 - val_accuracy: 0.7544 - val_loss: 0.6160",
"accuracy: 0.9334 - loss: 0.1840 - val_accuracy: 0.7797 - val_loss: 0.6251"
]

lstm_64_250_2 = [
"accuracy: 0.7601 - loss: 0.4893 - val_accuracy: 0.8142 - val_loss: 0.5053",
"accuracy: 0.9005 - loss: 0.2611 - val_accuracy: 0.8071 - val_loss: 0.5727",
"accuracy: 0.9191 - loss: 0.2189 - val_accuracy: 0.7981 - val_loss: 0.5938"
]

lstm_64_250_3 = [
"accuracy: 0.7668 - loss: 0.4790 - val_accuracy: 0.7475 - val_loss: 0.8763",
"accuracy: 0.8890 - loss: 0.2853 - val_accuracy: 0.8528 - val_loss: 0.4414",
"accuracy: 0.9093 - loss: 0.2369 - val_accuracy: 0.7042 - val_loss: 0.8548",
"accuracy: 0.9295 - loss: 0.1916 - val_accuracy: 0.7458 - val_loss: 0.7561"
]

lstm_64_250_4 = [
"accuracy: 0.7651 - loss: 0.4848 - val_accuracy: 0.7790 - val_loss: 0.5414",
"accuracy: 0.8980 - loss: 0.2625 - val_accuracy: 0.7754 - val_loss: 0.5856",
"accuracy: 0.9131 - loss: 0.2263 - val_accuracy: 0.8320 - val_loss: 0.4304",
"accuracy: 0.9335 - loss: 0.1795 - val_accuracy: 0.7491 - val_loss: 0.6729",
"accuracy: 0.9469 - loss: 0.1481 - val_accuracy: 0.6735 - val_loss: 1.1593"
]

lstm_64_250_5 = [
"accuracy: 0.7620 - loss: 0.4828 - val_accuracy: 0.8396 - val_loss: 0.4701",
"accuracy: 0.8556 - loss: 0.3495 - val_accuracy: 0.7879 - val_loss: 0.5197",
"accuracy: 0.9119 - loss: 0.2311 - val_accuracy: 0.8035 - val_loss: 0.4920"
]

lstm_128_500_1 = [
"accuracy: 0.7375 - loss: 0.5174 - val_accuracy: 0.7139 - val_loss: 0.6619",
"accuracy: 0.8950 - loss: 0.2635 - val_accuracy: 0.7875 - val_loss: 0.5613",
"accuracy: 0.9116 - loss: 0.2252 - val_accuracy: 0.7714 - val_loss: 0.6170",
"accuracy: 0.9345 - loss: 0.1835 - val_accuracy: 0.7985 - val_loss: 0.5326",
"accuracy: 0.9430 - loss: 0.1583 - val_accuracy: 0.7372 - val_loss: 0.7284"
]

lstm_128_500_2 = [
"accuracy: 0.7380 - loss: 0.5173 - val_accuracy: 0.8631 - val_loss: 0.3678",
"accuracy: 0.8858 - loss: 0.2884 - val_accuracy: 0.8489 - val_loss: 0.4176",
"accuracy: 0.9108 - loss: 0.2289 - val_accuracy: 0.7829 - val_loss: 0.6345"
]

lstm_128_500_3 = [
"accuracy: 0.7326 - loss: 0.5146 - val_accuracy: 0.6246 - val_loss: 0.9302",
"accuracy: 0.8966 - loss: 0.2662 - val_accuracy: 0.8166 - val_loss: 0.5902",
"accuracy: 0.9157 - loss: 0.2271 - val_accuracy: 0.7533 - val_loss: 0.7390",
"accuracy: 0.9211 - loss: 0.2113 - val_accuracy: 0.7997 - val_loss: 0.5532",
"accuracy: 0.9402 - loss: 0.1682 - val_accuracy: 0.7199 - val_loss: 0.7964"
]

lstm_128_500_4 = [
"accuracy: 0.7471 - loss: 0.5057 - val_accuracy: 0.7339 - val_loss: 0.6656",
"accuracy: 0.8997 - loss: 0.2574 - val_accuracy: 0.7216 - val_loss: 0.7188",
"accuracy: 0.9213 - loss: 0.2089 - val_accuracy: 0.6804 - val_loss: 0.9964"
]

lstm_128_500_5 = [
"accuracy: 0.7368 - loss: 0.5118 - val_accuracy: 0.7616 - val_loss: 0.5723",
"accuracy: 0.8953 - loss: 0.2609 - val_accuracy: 0.8301 - val_loss: 0.4524",
"accuracy: 0.9160 - loss: 0.2203 - val_accuracy: 0.8274 - val_loss: 0.4510",
"accuracy: 0.9267 - loss: 0.1986 - val_accuracy: 0.8406 - val_loss: 0.4470",
"accuracy: 0.9353 - loss: 0.1789 - val_accuracy: 0.8120 - val_loss: 0.5511"
]

def extract_metrics(data):
  metrics = []
  for entry in data:
    match = re.search(r"accuracy: (\d+\.\d+) - loss: (\d+\.\d+) - val_accuracy: (\d+\.\d+) - val_loss: (\d+\.\d+)", entry)
    if match:
      accuracy = float(match.group(1))
      loss = float(match.group(2))
      val_accuracy = float(match.group(3))
      val_loss = float(match.group(4))
      metrics.append((accuracy, loss, val_accuracy, val_loss))
  return metrics

lstm_64_250_1_metrics = extract_metrics(lstm_64_250_1)
lstm_64_250_2_metrics = extract_metrics(lstm_64_250_2)
lstm_64_250_3_metrics = extract_metrics(lstm_64_250_3)
lstm_64_250_4_metrics = extract_metrics(lstm_64_250_4)
lstm_64_250_5_metrics = extract_metrics(lstm_64_250_5)

# print(lstm_64_250_1_metrics)
# print(lstm_64_250_2_metrics)
# print(lstm_64_250_3_metrics)
# print(lstm_64_250_4_metrics)
# print(lstm_64_250_5_metrics)

lstm_128_500_1_metrics = extract_metrics(lstm_128_500_1)
lstm_128_500_2_metrics = extract_metrics(lstm_128_500_2)
lstm_128_500_3_metrics = extract_metrics(lstm_128_500_3)
lstm_128_500_4_metrics = extract_metrics(lstm_128_500_4)
lstm_128_500_5_metrics = extract_metrics(lstm_128_500_5)

# print(lstm_128_500_1_metrics)
# print(lstm_128_500_2_metrics)
# print(lstm_128_500_3_metrics)
# print(lstm_128_500_4_metrics)
# print(lstm_128_500_5_metrics)

accuracy_epochs = [0,0,0,0,0]
loss_epochs = [0,0,0,0,0]
val_accuracy_epochs = [0,0,0,0,0]
val_loss_epochs = [0,0,0,0,0]

epochs_counter = [0,0,0,0,0]

lstm_64_250_metrics = [lstm_64_250_1_metrics, lstm_64_250_2_metrics, lstm_64_250_3_metrics, lstm_64_250_4_metrics, lstm_64_250_5_metrics]

lstm_128_500_metrics = [lstm_128_500_1_metrics,
                        lstm_128_500_2_metrics,
                        lstm_128_500_3_metrics,
                        lstm_128_500_4_metrics,
                        lstm_128_500_5_metrics]

# for fold in lstm_64_250_metrics:
for fold in lstm_128_500_metrics:
  for index, metrics in enumerate(fold):
    print(f"{index} - {metrics}")
    accuracy_epochs[index] += metrics[0]
    loss_epochs[index] += metrics[1]
    val_accuracy_epochs[index] += metrics[2]
    val_loss_epochs[index] += metrics[3]

    epochs_counter[index] += 1

print("---------------")

print("accuracy_epochs")
print(accuracy_epochs)
print("loss_epochs")
print(loss_epochs)
print("val_accuracy_epochs")
print(val_accuracy_epochs)
print("val_loss_epochs")
print(val_loss_epochs)
print("---------------")
for index, acc in enumerate(accuracy_epochs): accuracy_epochs[index] = acc/epochs_counter[index]
for index, loss in enumerate(loss_epochs): loss_epochs[index] = loss/epochs_counter[index]
for index, acc in enumerate(val_accuracy_epochs): val_accuracy_epochs[index] = acc/epochs_counter[index]
for index, loss in enumerate(val_loss_epochs): val_loss_epochs[index] = loss/epochs_counter[index]


print("accuracy_epochs")
print(accuracy_epochs)
print("loss_epochs")
print(loss_epochs)
print("val_accuracy_epochs")
print(val_accuracy_epochs)
print("val_loss_epochs")
print(val_loss_epochs)

accuracy_epochs.insert(0, 0)
loss_epochs.insert(0, 1)

val_accuracy_epochs.insert(0,0)
val_loss_epochs.insert(0, 1)

epochs = list(range(len(accuracy_epochs)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(epochs, accuracy_epochs, linestyle='-', color='b', label="Training Accuracy")
ax1.plot(epochs, loss_epochs, linestyle='-', color='r', label="Training Loss")
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy/Loss')
ax1.set_title('Training Accuracy and Loss over Epochs')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
ax1.legend()

ax2.plot(epochs, val_accuracy_epochs, linestyle='-', color='b', label="Validation Accuracy")
ax2.plot(epochs, val_loss_epochs, linestyle='-', color='r', label="Validation Loss")
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy/Loss')
ax2.set_title('Validation Accuracy and Loss over Epochs')
ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
ax2.legend()

plt.subplots_adjust(wspace=5)
plt.tight_layout()
plt.show()

# plt.plot(epochs, accuracy_epochs, linestyle='-', color='b', label="accuracy")
# plt.plot(epochs, loss_epochs, linestyle='-', color='r', label="loss")
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Accuracy over Epochs')
# plt.legend()
# plt.show()

