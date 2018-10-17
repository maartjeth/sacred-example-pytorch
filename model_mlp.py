########################################################################################################################
#                                                                                                                      #
# The main body of this code is taken from:                                                                            #
# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/01-basics/feedforward_neural_network                #
#                                                                                                                      #
# Adaptations by Maartje ter Hoeve.                                                                                    #
# Comments about adaptations specifically to run this code with Sacred start with 'SACRED'                             #
#                                                                                                                      #
# Please have a look at the Sacred documentations for full details about Sacred itself: https://sacred.readthedocs.io/ #
#                                                                                                                      #
########################################################################################################################

import torch.nn as nn


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
