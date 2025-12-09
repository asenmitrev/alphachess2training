"""
Neural network model for AlphaZero.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaZeroNet(nn.Module):
    """
    Neural network for AlphaZero.
    Takes game state as input and outputs:
    - Policy: probability distribution over actions
    - Value: expected outcome from current player's perspective
    """
    
    def __init__(self, board_size: int = 5, num_channels: int = 64, num_residual_blocks: int = 3):
        """
        Initialize the network.
        
        Args:
            board_size: Size of the board (5 for King Capture)
            num_channels: Number of channels in convolutional layers
            num_residual_blocks: Number of residual blocks
        """
        super(AlphaZeroNet, self).__init__()
        self.board_size = board_size
        self.num_channels = num_channels
        self.num_residual_blocks = num_residual_blocks
        # Action space: 2 pieces * board_size * board_size
        self.action_size = 2 * board_size * board_size
        
        # Input: 1 channel (canonical state)
        # Initial convolution
        self.conv = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_channels)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_residual_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)
        
        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, num_channels)
        self.value_fc2 = nn.Linear(num_channels, 1)
        
        # Initialize value head final layer with small weights to prevent tanh saturation
        # This ensures gradients can flow and the model can learn fine-grained value distinctions
        nn.init.uniform_(self.value_fc2.weight, -0.01, 0.01)
        nn.init.constant_(self.value_fc2.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, board_size, board_size)
        
        Returns:
            policy: Policy logits of shape (batch_size, action_size)
            value: Value estimate of shape (batch_size, 1)
        """
        # Initial convolution
        x = F.relu(self.bn(self.conv(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value


class ResidualBlock(nn.Module):
    """Residual block for the neural network."""
    
    def __init__(self, num_channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

