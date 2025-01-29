import re
import numpy as np
import matplotlib.pyplot as plt

lstm_64_250_1 = [
"accuracy: 0.7664 - loss: 0.4729 - val_accuracy: 0.7761 - val_loss: 0.5243",
"accuracy: 0.9013 - loss: 0.2552 - val_accuracy: 0.7700 - val_loss: 0.5867",
"accuracy: 0.9212 - loss: 0.2137 - val_accuracy: 0.7479 - val_loss: 0.5987"
]

lstm_64_250_2 = [
"accuracy: 0.7694 - loss: 0.4714 - val_accuracy: 0.7804 - val_loss: 0.6751","accuracy: 0.8992 - loss: 0.2620 - val_accuracy: 0.8438 - val_loss: 0.3806","accuracy: 0.9163 - loss: 0.2182 - val_accuracy: 0.8921 - val_loss: 0.2717","accuracy: 0.9297 - loss: 0.1879 - val_accuracy: 0.8319 - val_loss: 0.4650","accuracy: 0.9445 - loss: 0.1553 - val_accuracy: 0.8234 - val_loss: 0.5362"
]

lstm_64_250_3 = [
"accuracy: 0.7578 - loss: 0.4897 - val_accuracy: 0.8291 - val_loss: 0.4382","accuracy: 0.8975 - loss: 0.2590 - val_accuracy: 0.7715 - val_loss: 0.5689","accuracy: 0.9205 - loss: 0.2147 - val_accuracy: 0.8036 - val_loss: 0.5620"
]

lstm_64_250_4 = [
"accuracy: 0.7682 - loss: 0.4719 - val_accuracy: 0.6814 - val_loss: 0.7280","accuracy: 0.8925 - loss: 0.2748 - val_accuracy: 0.7744 - val_loss: 0.5880","accuracy: 0.9194 - loss: 0.2111 - val_accuracy: 0.7605 - val_loss: 0.5986","accuracy: 0.9314 - loss: 0.1819 - val_accuracy: 0.7686 - val_loss: 0.6913"
]

lstm_64_250_5 = [
"accuracy: 0.7700 - loss: 0.4688 - val_accuracy: 0.7713 - val_loss: 0.5759","accuracy: 0.9017 - loss: 0.2538 - val_accuracy: 0.8298 - val_loss: 0.3819","accuracy: 0.9192 - loss: 0.2103 - val_accuracy: 0.8357 - val_loss: 0.4081","accuracy: 0.9311 - loss: 0.1800 - val_accuracy: 0.7896 - val_loss: 0.5853"
]

lstm_128_500_1 = ["accuracy: 0.7391 - loss: 0.5046 - val_accuracy: 0.7550 - val_loss: 0.5538","accuracy: 0.9022 - loss: 0.2530 - val_accuracy: 0.7426 - val_loss: 0.5815","accuracy: 0.9243 - loss: 0.2064 - val_accuracy: 0.7470 - val_loss: 0.6874"]
lstm_128_500_2 = [
"accuracy: 0.7404 - loss: 0.5118 - val_accuracy: 0.7680 - val_loss: 0.5330","accuracy: 0.8981 - loss: 0.2680 - val_accuracy: 0.7384 - val_loss: 0.6051","accuracy: 0.9131 - loss: 0.2379 - val_accuracy: 0.7729 - val_loss: 0.6031"]
lstm_128_500_3 = [
"accuracy: 0.7415 - loss: 0.5081 - val_accuracy: 0.7566 - val_loss: 0.5837","accuracy: 0.8951 - loss: 0.2693 - val_accuracy: 0.7408 - val_loss: 0.7683","accuracy: 0.9175 - loss: 0.2211 - val_accuracy: 0.7406 - val_loss: 0.7184"
]
lstm_128_500_4 = [
"accuracy: 0.7438 - loss: 0.4995 - val_accuracy: 0.8192 - val_loss: 0.4522","accuracy: 0.8996 - loss: 0.2557 - val_accuracy: 0.7993 - val_loss: 0.5270","accuracy: 0.9182 - loss: 0.2162 - val_accuracy: 0.7444 - val_loss: 0.8364"
]
lstm_128_500_5 = [
"accuracy: 0.7445 - loss: 0.4980 - val_accuracy: 0.7934 - val_loss: 0.5112","accuracy: 0.9022 - loss: 0.2524 - val_accuracy: 0.8516 - val_loss: 0.3550","accuracy: 0.9182 - loss: 0.2234 - val_accuracy: 0.8120 - val_loss: 0.4818","accuracy: 0.9292 - loss: 0.1901 - val_accuracy: 0.7757 - val_loss: 0.5957"
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

for fold in lstm_64_250_metrics:
  for index, metrics in enumerate(fold):
    print(f"{index} - {metrics}")
    accuracy_epochs[index] += metrics[0]
    loss_epochs[index] += metrics[1]
    val_accuracy_epochs[index] += metrics[2]
    val_loss_epochs[index] += metrics[3]

    epochs_counter[index] += 1

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
print(accuracy_epochs)
print("val_loss_epochs")
print(loss_epochs)

epochs = list(range(1, len(accuracy_epochs) + 1))
plt.plot(epochs, accuracy_epochs, linestyle='-', color='b')
plt.plot(epochs, loss_epochs, linestyle='-', color='r')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.show()