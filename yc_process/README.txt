*********************************
Dataloader based on Ninapro.py(From WY Guo) and Ninapro_dataset.py(from NinaproNet)

*********************************
network runing now
Input: tensor[64, 200, 12]
64 is the batch size.
200 is the sequence length.
12 is the dimensionality of each element in the sequence.(probably 36 in new version since the low_filter_3ch(emgs,fs) in dataloader.

current train accu : 0.66
current valid accu : 0.55
only input Exercise 1 with 18 class number
window size: 200
base feature: 64
