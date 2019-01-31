# frame-predict
Predicting image frames using LSTM + CNN with pytorch

Dataset is sequence of falling dot
![alt text](https://raw.githubusercontent.com/marisancans/frame-predict/master/Figure_1.png)

Given sequence of frames, the desired output is the next frame.
Network consists of few convolution layers to get image features that are later passed to LSTM or RNN layer.
Loss function is pixel subtraction between actual and predicted image.

![alt text](https://raw.githubusercontent.com/marisancans/frame-predict/master/graph.png)
Taken from: 
Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning
[https://arxiv.org/abs/1605.08104]
