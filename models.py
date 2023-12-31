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
        self.apply(utils.weight_init)

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

    def forward(self, z, action):
        x = torch.cat([z, action], axis=-1)
        x = self.model(x)
        mean, std = _get_mean_std(x, self.std_min, self.std_max)
        mean += z
        return _independent_normal_mean_std(mean, std)


class ModelContrafactualPrior(nn.Module):
    def __init__(self, latent_dims, action_dims, hidden_dims, num_layers=2):
        super().__init__()
        self.latent_dims = latent_dims
        self.action_dims = action_dims
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.std_min = 0.1 # TODO tune? seems important hyperparam
        self.std_max = 10
        self.independent_change_model = self._build_ic_model()
        self.effect_contrafactual_model = self._build_ec_model()
        self.apply(utils.weight_init)

    def _build_ic_model(self):
        model = [nn.Linear(self.latent_dims, self.hidden_dims)]
        model += [nn.ELU()]
        for i in range(self.num_layers-1):
            model += [nn.Linear(self.hidden_dims, self.hidden_dims)]
            model += [nn.ELU()]
        model += [nn.Linear(self.hidden_dims, 2*self.latent_dims)]
        return nn.Sequential(*model)

    def _build_ec_model(self):
        hidden_dims = self.hidden_dims // 2  # TODO tune
        model = [nn.Linear(self.action_dims + self.latent_dims, hidden_dims)]
        model += [nn.ELU()]
        for i in range(self.num_layers-1):
            model += [nn.Linear(hidden_dims, hidden_dims)]
            model += [nn.ELU()]
        model += [nn.Linear(hidden_dims, 2*self.latent_dims)]
        return nn.Sequential(*model)

    def calculate_diff(self, z, action):
        return _independent_normal_mean_std(*self._calculate_diff_mean_std(z, action))

    def _calculate_diff_mean_std(self, z, action):
        x = torch.cat((z, action), dim=-1)
        ec_mean, ec_std = self._get_mean_std(self.effect_contrafactual_model(x))
        ic_mean, ic_std = self._get_mean_std(self.independent_change_model(z))
        return ec_mean+ic_mean, ec_std + ic_std

    def ic(self, z):
        return _independent_normal_mean_std(*self._get_mean_std(self.independent_change_model(z)))

    def ec(self, z, a):
        return _independent_normal_mean_std(*self._get_mean_std(self.effect_contrafactual_model(z, a)))

    def forward(self, actual_z, actual_action, next_actual_z, contr_z, contr_action):
        # mean, std = self._calculate_diff_mean_std(contr_z, contr_action)
        # mean += contr_z
        # return _independent_normal(mean, std)
        if actual_z is contr_z:
            z_a_act = torch.cat((actual_z, actual_action), dim=-1)
            actual_effect, std_a = self._get_mean_std(self.effect_contrafactual_model(z_a_act))
            z_a_contr = torch.cat((contr_z, contr_action), dim=-1)
            contr_effect, std_c = self._get_mean_std(self.effect_contrafactual_model(z_a_contr))
        else:
            actual_effect, std_a = self._calculate_diff_mean_std(actual_z, actual_action)
            contr_effect, std_c = self._calculate_diff_mean_std(contr_z, contr_action)

        mean = next_actual_z - actual_z + contr_z - actual_effect + contr_effect
        return _independent_normal_mean_std(mean, self.std_min)


class RewardPrior(nn.Module):
    def __init__(self, latent_dims, hidden_dims, action_dims):
        super().__init__()
        self.reward = nn.Sequential(
            nn.Linear(latent_dims + action_dims, hidden_dims), nn.LayerNorm(hidden_dims), 
            nn.Tanh(), nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(), nn.Linear(hidden_dims, 1))
        self.apply(utils.weight_init)
        
    def forward(self, z, a):
        z_a = torch.cat([z, a], -1)
        reward = self.reward(z_a)
        return reward


class RewardStateActionPrior(nn.Module):
    def __init__(self, latent_dims, hidden_dims, action_dims):
        super().__init__()
        self.reward_state = nn.Sequential(
            nn.Linear(latent_dims, hidden_dims), nn.LayerNorm(hidden_dims),
            nn.Tanh(), nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(), nn.Linear(hidden_dims, 1))
        hidden_dims //= 2  # TODO tune
        self.reward_action = nn.Sequential(
            nn.Linear(latent_dims + action_dims, hidden_dims), nn.LayerNorm(hidden_dims),
            nn.Tanh(), nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(), nn.Linear(hidden_dims, 1))
        self.apply(utils.weight_init)


    def state_reward(self, z):
        return self.reward_state(z)

    def state_action_reward(self, z, a):
        z_a = torch.cat([z, a], -1)
        return self.reward_action(z_a)

    def calculate_diff(self, z, a):
        return self.state_reward(z) + self.state_action_reward(z, a)

    def forward(self, actual_reward, actual_z, actual_action, contr_z, contr_action):
        # TODO check if there would be a difference if I subtracted the whole trajectory effect and
        # added whole contrafactual effect (I think it matters)
        return actual_reward \
            - self.state_reward(actual_z) \
            - self.state_action_reward(actual_z, actual_action) \
            + self.state_reward(contr_z) \
            + self.state_action_reward(contr_z, contr_action)

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
        
class Critic(nn.Module):
    def __init__(self, latent_dims, hidden_dims, action_shape):
        super().__init__()
        self.Q1 = nn.Sequential(
            nn.Linear(latent_dims + action_shape, hidden_dims), nn.LayerNorm(hidden_dims), 
            nn.Tanh(), nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(), nn.Linear(hidden_dims, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(latent_dims + action_shape, hidden_dims), nn.LayerNorm(hidden_dims), 
            nn.Tanh(), nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(), nn.Linear(hidden_dims, 1))
            
        self.apply(utils.weight_init)

    def forward(self, x, a):
        x_a = torch.cat([x, a], -1)
        q1 = self.Q1(x_a)
        q2 = self.Q2(x_a)
        return q1, q2


class ModelCritic(Critic):
    def __init__(self, latent_dims, hidden_dims, action_shape, gamma, model, reward):
        super().__init__(latent_dims, hidden_dims, action_shape=0)
        self.model = model
        self.reward = reward
        self.gamma = gamma
    def forward(self, x, a):
        with utils.FreezeParameters([self.reward, self.model]):
            next_state = self.model(x, a)
            R = self.reward(x, a)
        q1 = self.Q1(next_state.rsample())
        q2 = self.Q2(next_state.rsample())
        return q1*self.gamma + R, q2*self.gamma + R

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
