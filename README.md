# Apple-diseaseAI

# Project Name
Considering the development of agriculture, apple diseases can affect the yield of apples and food security to some extent.
This classifier is able to classify 2 different diseases(apple scab & black rot) and healthy apple based on the given pictures of apple leaves. With the results of the algorithm, we can have a clear idea on the growth of apples and take proper actions more quickly. Therefore we can predict yields and reduce losses more efficiently.


# The Algorithm:
The algorithm works as a classification neural network, we use transfer learning to train the resnet-18 based imagenet classifier. We ran training on 7400 pictures of these 3 kinds of apple leaves and reached the acuuracy of 92% eventually.

dataset link: 
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

resnet18.onnx: 
https://drive.google.com/file/d/1gw_F9eIYatnb6WoZY3Z3oO0dewnUTLWN/view?usp=sharing

This is an example picture of a leaf with apple scab

![scab0001](https://github.com/user-attachments/assets/450e95fc-70fb-4915-bd8c-bb889f744ada)


This image shows that the AI is 97.7% sure it has apple scab)
![image](https://github.com/user-attachments/assets/26d55b85-2428-4cda-838a-6333660aab73)




# Run this Project:

1.Navigate to the /home/nvidia/jetson-inference/python/training/classification

2.Set bash enviorment variables "NET=models/Apple_disease_classification" and "DATASET = data/Apple_disease_classification"

3.Run the command "imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/(Apple_apple_scab/Apple_healthy/Apple_black_rot)/(imageofappleleavesfromtestdata.JPG) (nameofwhereitwillbesaved.JPG)"

4.The file will now appeare in /home/nvidia/jetson-inference/python/training/classification




