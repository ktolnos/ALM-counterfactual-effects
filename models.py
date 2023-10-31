import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F

import utils


def _get_mean_std(x, std_min, std_max):
    mean, std = torch.chunk(x, 2, -1)
    mean = 30 * torch.tanh(mean / 30)
    std = std_max - F.softplus(std_max - std)
    std = std_min + F.softplus(std - std_min)
    return mean, std


def _independent_normal_mean_std(mean, std):
    return td.independent.Independent(td.Normal(mean, std), 1)


def _independent_normal_x(x, std_min, std_max):
    return _independent_normal_mean_std(*_get_mean_std(x, std_min, std_max))

class Encoder(nn.Module):
    def __init__(self, input_shape, hidden_dims, latent_dims, min_std, max_std):
        super().__init__()
        self.latent_dims = latent_dims
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, hidden_dims),
            nn.ELU(), nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(), nn.Linear(hidden_dims, 2*latent_dims))

        self.std_min = min_std
        self.std_max = max_std
        self.apply(utils.weight_init)

    def forward(self, x):
        x = self.encoder(x)
        return _independent_normal_x(x, self.std_min, self.std_max)

class ConvEncoder(nn.Module):
    def __init__(self, input_shape, hidden_dims, latent_dims, min_std, max_std):
        super().__init__()
        self.latent_dims = latent_dims
        self.encoder = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        nn.ELU(),
        nn.Linear(900, 500))


        self.std_min = min_std
        self.std_max = max_std
        self.apply(utils.weight_init)

    def forward(self, x):
        x = self.encoder(x)
        return _independent_normal_x(x, self.std_min, self.std_max)
class RepModelPrior(nn.Module):
    def __init__(self, latent_dims, min_std, max_std, init_vector):

        super().__init__()
        self.latent_dims = latent_dims
        self.std_min = min_std
        self.std_max = max_std
        self.h = None
        
        self.model = nn.GRUCell(
            input_size=self.latent_dims, 
            hidden_size=self.latent_dims*2,
            bias=True
        )
        
        # Linear layer to map from latent_dims*2 to latent_dims
        self.fc = nn.Linear(latent_dims*2, latent_dims*2)
    def forward(self, z):
        # Pass the input through the RNN
        self.h = self.h.detach()

        self.h = self.model(z, self.h)
        
        z = self.fc(self.h) 


    
        return _independent_normal_x(z, self.std_min, self.std_max) 

    def reset(self):
        self.h = None  

class RepModelDiffPrior(RepModelPrior):
    def __init__(self, latent_dims, min_std, max_std, init_vector):

        super().__init__(latent_dims, min_std, max_std, init_vector)

    def forward(self, z):
        if self.h is not None:
            self.h = self.h.detach()

        else:
            self.h = torch.zeros(z.shape[0], 2 * z.shape[1], device=z.device)
        dz = self.model(z, self.h)
        dz = self.fc(dz)
        mean, std = _get_mean_std(dz, self.std_min, self.std_max)
        z = z + mean
        return _independent_normal_mean_std(z, std)



class ModelPrior(nn.Module):
    def __init__(self, latent_dims, action_dims, hidden_dims, min_std, max_std, num_layers=2):
        super().__init__()
        self.latent_dims = latent_dims
        self.action_dims = action_dims
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.std_min = min_std
        self.std_max = max_std
        self.model = self._build_model()

    def _build_model(self):
        model = [nn.Linear(self.action_dims + self.latent_dims, self.hidden_dims)]
        model += [nn.ELU()]
        for i in range(self.num_layers-1):
            model += [nn.Linear(self.hidden_dims, self.hidden_dims)]
            model += [nn.ELU()]
        model += [nn.Linear(self.hidden_dims, 2*self.latent_dims)]
        return nn.Sequential(*model)

    def forward(self, z, action):
        x = torch.cat([z, action], axis=-1)
        x = self.model(x)
        return _independent_normal_x(x, self.std_min, self.std_max)


class ModelDiffPrior(ModelPrior):
    def __init__(self, latent_dims, action_dims, hidden_dims, min_std, max_std, num_layers=2):
        super().__init__(latent_dims, action_dims, hidden_dims, min_std, max_std, num_layers)
    def forward(self, z, action):
        x = torch.cat([z, action], axis=-1)
        x = self.model(x)
        mean, std = _get_mean_std(x, self.std_min, self.std_max)
        mean += z
        return _independent_normal_mean_std(mean, std)


class RewardPrior(nn.Module):
    def __init__(self, latent_dims, hidden_dims, action_dims):
        super().__init__()
        self.reward_action = nn.Sequential(
            nn.Linear(latent_dims + action_dims, hidden_dims), nn.LayerNorm(hidden_dims),
            nn.Tanh(), nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(), nn.Linear(hidden_dims, 1))
        self.apply(utils.weight_init)


    def state_action_reward(self, z, a):
        z_a = torch.cat([z, a], -1)
        return self.reward_action(z_a)

    def calculate_diff(self, z, a):
        return self.state_reward(z) + self.state_action_reward(z, a)

    def forward(self, actual_z, actual_action):
        return self.state_action_reward(actual_z, actual_action) 

class Discriminator(nn.Module):
    def __init__(self, latent_dims, hidden_dims, action_dims):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2 * latent_dims + action_dims, hidden_dims), nn.LayerNorm(hidden_dims),
            nn.Tanh(), nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(), nn.Linear(hidden_dims, 2))
        self.apply(utils.weight_init)

    def forward(self, z, a, z_next):
        x = torch.cat([z, a, z_next], -1)
        logits = self.classifier(x)
        return logits
    
    def get_reward(self, z, a, z_next):
        x = torch.cat([z, a, z_next], -1)
        logits = self.classifier(x)
        reward = torch.sub(logits[..., 1], logits[..., 0])
        return reward.unsqueeze(-1)
        
class RepDiscriminator(Discriminator):
    def __init__(self, latent_dims, hidden_dims):
        super().__init__(latent_dims, hidden_dims, 0)
        self.classifier = nn.Sequential(
            nn.Linear(2 * latent_dims, hidden_dims), nn.LayerNorm(hidden_dims),
            nn.Tanh(), nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(), nn.Linear(hidden_dims, 2))
        self.apply(utils.weight_init)

    def forward(self, z1, z2):
        x = torch.cat([z1, z2], -1)
        logits = self.classifier(x)
        return logits
    
    def get_reward(self, z1, z2):
        x = torch.cat([z1, z2], -1)
        logits = self.classifier(x)
        reward = torch.sub(logits[..., 1], logits[..., 0])
        return reward.unsqueeze(-1)
        


class Critic(nn.Module):
    def __init__(self, latent_dims, hidden_dims, action_shape, model, reward):
        super().__init__()
        self.Q1 = nn.Sequential(
            nn.Linear(latent_dims, hidden_dims), nn.LayerNorm(hidden_dims), 
            nn.Tanh(), nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(), nn.Linear(hidden_dims, 1))
        self.model = model
        self.reward = reward
        self.Q2 = nn.Sequential(
            nn.Linear(latent_dims, hidden_dims), nn.LayerNorm(hidden_dims), 
            nn.Tanh(), nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(), nn.Linear(hidden_dims, 1))
            
        self.apply(utils.weight_init)

    def forward(self, x, a):
        next_state = self.model(x, a)
        R = self.reward(x, a)
        q1 = self.Q1(next_state.rsample())
        q2 = self.Q2(next_state.rsample())
        return q1, q2, R

class Actor(nn.Module):
    def __init__(self, input_shape, hidden_dims, output_shape, low, high):
        super(Actor, self).__init__()

        self.low = low
        self.high = high
        self.fc1 = nn.Linear(input_shape, hidden_dims) 
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.mean = nn.Linear(hidden_dims, output_shape)
        self.apply(utils.weight_init)

    def forward(self, x, std):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        mean = torch.tanh(self.mean(x))
        std = torch.ones_like(mean) * std
        dist = utils.TruncatedNormal(mean, std, self.low, self.high)
        return dist
        
class StochasticActor(nn.Module):
    def __init__(self, input_shape, hidden_dims, output_shape, low, high):
        super(StochasticActor, self).__init__()
        self.low = low
        self.high = high
        self.fc1 = nn.Linear(input_shape, hidden_dims) 
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, 2*output_shape)
        self.std_min = np.exp(-5)
        self.std_max = np.exp(2)
        self.apply(utils.weight_init)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        mean, std = torch.chunk(x, 2, -1)
        mean = torch.tanh(mean)
        std = self.std_max - F.softplus(self.std_max-std)
        std = self.std_min  + F.softplus(std-self.std_min) 
        dist = utils.TruncatedNormal(mean, std, self.low, self.high)
        return dist
