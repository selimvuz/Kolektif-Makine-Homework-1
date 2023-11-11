import re

text = """
Epoch 1/50
161/161 [==============================] - 5s 28ms/step - loss: 1.0978 - accuracy: 0.3656 - val_loss: 1.0963 - val_accuracy: 0.3841
Epoch 2/50
161/161 [==============================] - 5s 28ms/step - loss: 1.0953 - accuracy: 0.3693 - val_loss: 1.0941 - val_accuracy: 0.3841
Epoch 3/50
161/161 [==============================] - 5s 29ms/step - loss: 1.0932 - accuracy: 0.3693 - val_loss: 1.0922 - val_accuracy: 0.3841
Epoch 4/50
161/161 [==============================] - 5s 29ms/step - loss: 1.0914 - accuracy: 0.3693 - val_loss: 1.0907 - val_accuracy: 0.3841
Epoch 5/50
161/161 [==============================] - 5s 29ms/step - loss: 1.0900 - accuracy: 0.3693 - val_loss: 1.0895 - val_accuracy: 0.3841
Epoch 6/50
161/161 [==============================] - 5s 29ms/step - loss: 1.0888 - accuracy: 0.3693 - val_loss: 1.0885 - val_accuracy: 0.3841
Epoch 7/50
161/161 [==============================] - 5s 28ms/step - loss: 1.0878 - accuracy: 0.3693 - val_loss: 1.0876 - val_accuracy: 0.3841
Epoch 8/50
161/161 [==============================] - 5s 29ms/step - loss: 1.0870 - accuracy: 0.3706 - val_loss: 1.0869 - val_accuracy: 0.3850
Epoch 9/50
161/161 [==============================] - 5s 30ms/step - loss: 1.0862 - accuracy: 0.3799 - val_loss: 1.0862 - val_accuracy: 0.3902
Epoch 10/50
161/161 [==============================] - 5s 30ms/step - loss: 1.0856 - accuracy: 0.3835 - val_loss: 1.0857 - val_accuracy: 0.4156
Epoch 11/50
161/161 [==============================] - 5s 29ms/step - loss: 1.0850 - accuracy: 0.4073 - val_loss: 1.0852 - val_accuracy: 0.4366
Epoch 12/50
161/161 [==============================] - 4s 28ms/step - loss: 1.0845 - accuracy: 0.4238 - val_loss: 1.0847 - val_accuracy: 0.4436
Epoch 13/50
161/161 [==============================] - 4s 27ms/step - loss: 1.0840 - accuracy: 0.4334 - val_loss: 1.0844 - val_accuracy: 0.4444
Epoch 14/50
161/161 [==============================] - 4s 28ms/step - loss: 1.0836 - accuracy: 0.4358 - val_loss: 1.0840 - val_accuracy: 0.4488
Epoch 15/50
161/161 [==============================] - 4s 27ms/step - loss: 1.0832 - accuracy: 0.4762 - val_loss: 1.0837 - val_accuracy: 0.4663
Epoch 16/50
161/161 [==============================] - 4s 28ms/step - loss: 1.0828 - accuracy: 0.4843 - val_loss: 1.0834 - val_accuracy: 0.4934
Epoch 17/50
161/161 [==============================] - 4s 27ms/step - loss: 1.0824 - accuracy: 0.5207 - val_loss: 1.0831 - val_accuracy: 0.5083
Epoch 18/50
161/161 [==============================] - 4s 28ms/step - loss: 1.0821 - accuracy: 0.5256 - val_loss: 1.0828 - val_accuracy: 0.5372
Epoch 19/50
161/161 [==============================] - 4s 27ms/step - loss: 1.0817 - accuracy: 0.5648 - val_loss: 1.0825 - val_accuracy: 0.5433
Epoch 20/50
161/161 [==============================] - 4s 28ms/step - loss: 1.0814 - accuracy: 0.5742 - val_loss: 1.0823 - val_accuracy: 0.5556
Epoch 21/50
161/161 [==============================] - 4s 28ms/step - loss: 1.0811 - accuracy: 0.5838 - val_loss: 1.0820 - val_accuracy: 0.5599
Epoch 22/50
161/161 [==============================] - 4s 28ms/step - loss: 1.0808 - accuracy: 0.5863 - val_loss: 1.0818 - val_accuracy: 0.5661
Epoch 23/50
161/161 [==============================] - 4s 28ms/step - loss: 1.0805 - accuracy: 0.5938 - val_loss: 1.0815 - val_accuracy: 0.5678
Epoch 24/50
161/161 [==============================] - 4s 28ms/step - loss: 1.0802 - accuracy: 0.5954 - val_loss: 1.0813 - val_accuracy: 0.5739
Epoch 25/50
161/161 [==============================] - 4s 28ms/step - loss: 1.0799 - accuracy: 0.5939 - val_loss: 1.0811 - val_accuracy: 0.5757
Epoch 26/50
161/161 [==============================] - 4s 28ms/step - loss: 1.0796 - accuracy: 0.5969 - val_loss: 1.0808 - val_accuracy: 0.5652
Epoch 27/50
161/161 [==============================] - 4s 28ms/step - loss: 1.0793 - accuracy: 0.5606 - val_loss: 1.0806 - val_accuracy: 0.5643
Epoch 28/50
161/161 [==============================] - 5s 28ms/step - loss: 1.0791 - accuracy: 0.5966 - val_loss: 1.0804 - val_accuracy: 0.5311
Epoch 29/50
161/161 [==============================] - 5s 29ms/step - loss: 1.0788 - accuracy: 0.5569 - val_loss: 1.0801 - val_accuracy: 0.5354
Epoch 30/50
161/161 [==============================] - 5s 28ms/step - loss: 1.0785 - accuracy: 0.5784 - val_loss: 1.0799 - val_accuracy: 0.5188
Epoch 31/50
161/161 [==============================] - 5s 28ms/step - loss: 1.0782 - accuracy: 0.5742 - val_loss: 1.0797 - val_accuracy: 0.5223
Epoch 32/50
161/161 [==============================] - 5s 28ms/step - loss: 1.0780 - accuracy: 0.5801 - val_loss: 1.0794 - val_accuracy: 0.5179
Epoch 33/50
161/161 [==============================] - 5s 28ms/step - loss: 1.0777 - accuracy: 0.5472 - val_loss: 1.0792 - val_accuracy: 0.5284
Epoch 34/50
161/161 [==============================] - 5s 28ms/step - loss: 1.0774 - accuracy: 0.5638 - val_loss: 1.0789 - val_accuracy: 0.5407
Epoch 35/50
161/161 [==============================] - 5s 28ms/step - loss: 1.0771 - accuracy: 0.5800 - val_loss: 1.0787 - val_accuracy: 0.5381
Epoch 36/50
161/161 [==============================] - 5s 29ms/step - loss: 1.0769 - accuracy: 0.5821 - val_loss: 1.0784 - val_accuracy: 0.5398
Epoch 37/50
161/161 [==============================] - 5s 29ms/step - loss: 1.0766 - accuracy: 0.5940 - val_loss: 1.0782 - val_accuracy: 0.5486
Epoch 38/50
161/161 [==============================] - 4s 27ms/step - loss: 1.0763 - accuracy: 0.5772 - val_loss: 1.0779 - val_accuracy: 0.5564
Epoch 39/50
161/161 [==============================] - 4s 27ms/step - loss: 1.0760 - accuracy: 0.5791 - val_loss: 1.0777 - val_accuracy: 0.5661
Epoch 40/50
161/161 [==============================] - 4s 27ms/step - loss: 1.0758 - accuracy: 0.5938 - val_loss: 1.0774 - val_accuracy: 0.5748
Epoch 41/50
161/161 [==============================] - 4s 28ms/step - loss: 1.0755 - accuracy: 0.5958 - val_loss: 1.0772 - val_accuracy: 0.5774
Epoch 42/50
161/161 [==============================] - 4s 27ms/step - loss: 1.0752 - accuracy: 0.5863 - val_loss: 1.0769 - val_accuracy: 0.5792
Epoch 43/50
161/161 [==============================] - 4s 27ms/step - loss: 1.0749 - accuracy: 0.5985 - val_loss: 1.0767 - val_accuracy: 0.5801
Epoch 44/50
161/161 [==============================] - 4s 28ms/step - loss: 1.0746 - accuracy: 0.5953 - val_loss: 1.0764 - val_accuracy: 0.5792
Epoch 45/50
161/161 [==============================] - 4s 27ms/step - loss: 1.0743 - accuracy: 0.5925 - val_loss: 1.0761 - val_accuracy: 0.5792
Epoch 46/50
161/161 [==============================] - 4s 27ms/step - loss: 1.0741 - accuracy: 0.5987 - val_loss: 1.0759 - val_accuracy: 0.5801
Epoch 47/50
161/161 [==============================] - 4s 27ms/step - loss: 1.0737 - accuracy: 0.5965 - val_loss: 1.0756 - val_accuracy: 0.5783
Epoch 48/50
161/161 [==============================] - 4s 27ms/step - loss: 1.0734 - accuracy: 0.5950 - val_loss: 1.0752 - val_accuracy: 0.5696
Epoch 49/50
161/161 [==============================] - 4s 28ms/step - loss: 1.0730 - accuracy: 0.5919 - val_loss: 1.0749 - val_accuracy: 0.5661
Epoch 50/50
161/161 [==============================] - 4s 28ms/step - loss: 1.0727 - accuracy: 0.5868 - val_loss: 1.0746 - val_accuracy: 0.5678
"""
# Initialize empty lists for accuracy and validation accuracy
accuracy_list = []
val_accuracy_list = []

# Split the text into lines and extract accuracy values
lines = text.strip().split('\n')
for line in lines:
    # Extract accuracy and validation accuracy using regular expressions
    match = re.search(r'accuracy: (\d+\.\d+).*val_accuracy: (\d+\.\d+)', line)
    if match:
        accuracy_list.append(float(match.group(1)))
        val_accuracy_list.append(float(match.group(2)))

# Print the lists
print("Accuracy List:", accuracy_list)
print("Validation Accuracy List:", val_accuracy_list)
