# frame-predict
Predicting image frames using LSTM + CNN with pytorch


Given n frames to the network, the desired output is next sequence of possible frames.
Network consists of few convolution layers to get image features that are later passed to LSTM or RNN layer.
Loss function is pixel subtraction between actual and predicted image.

