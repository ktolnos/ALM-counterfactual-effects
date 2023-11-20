import random
import time
from pathlib import Path

import numpy as np
import torch
import wandb

from utils.env import save_frames_as_gif
from workspaces.common import make_agent, make_env


class MujocoWorkspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.cfg = cfg
        if self.cfg.save_snapshot:
            self.checkpoint_path = self.work_dir / 'checkpoints'
            self.checkpoint_path.mkdir(exist_ok=True)
        self.device = torch.device(cfg.device)
        self.set_seed()
        self.train_env, self.eval_env, self.rollout_env = make_env(self.cfg)
        self.agent = make_agent(self.train_env, self.rollout_env, self.device, self.cfg)
        self._train_step = 0
        self._train_episode = 0
        self._best_eval_returns = -np.inf

    def set_seed(self):
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)

    def _explore(self):
        state, info = self.train_env.reset(seed=self.cfg.seed)
        
        for _ in range(1, self.cfg.explore_steps):
            action = self.train_env.action_space.sample()
            mujoco_state = self.train_env.sim.get_state()

            next_state, reward, done, trunc, info = self.train_env.step(action)

            self.agent.env_buffer.push((state, action, reward, next_state, done, trunc), mujoco_state)

            if done:
                state, info = self.train_env.reset(seed=self.cfg.seed)
                done = False
            else:
                state = next_state
            
    def train(self):
        self._explore()
        self._eval()
        state, info = self.train_env.reset(seed=self.cfg.seed)
        done = False
        episode_start_time = time.time()
        trains_per_action = 1
        ret = []
        val_seq= []
        rew = []
        trains_per_action = 1
        state_seq = []
        action_seq = []

        for _ in range(1, self.cfg.num_train_steps-self.cfg.explore_steps+1):  

            action = self.agent.get_action(state, self._train_step)
            mujoco_state = self.train_env.sim.get_state()
            next_state, reward, done, trunc, info = self.train_env.step(action)
            state_seq.append(state)
            action_seq.append(action)
            val_seq.append(self.agent.get_value(state, action)[0])
            rew.append(reward)

            done = done or trunc
            self._train_step += 1

            self.agent.env_buffer.push((state, action, reward, next_state, done, trunc), mujoco_state)
            val_seq.append(min(self.agent.get_value(state, action)))
            rew.append(reward)
            for i in range(trains_per_action):
                self.agent.update(self._train_step, i!=0)

            if (self._train_step)%self.cfg.eval_episode_interval==0:
                self._eval()

            if self.cfg.save_snapshot and (self._train_step)%self.cfg.save_snapshot_interval==0:
                self.save_snapshot()

            if done:
                ret = rew[:]
                ret[-1] = np.mean(rew[-100:])*1/(1-self.agent.gamma)

                for i in reversed(range(len(rew) - 1)):
                    ret[i] = rew[i] + self.agent.gamma * ret[i+1]
                xs = list(range(len(ret)))
                values = np.array(val_seq)
                print("Episode: {}, total numsteps: {}, return: {}".format(self._train_episode, self._train_step, round(info["episode"]["r"][0], 2)))
                self.agent.update_critic_offline(state_seq, action_seq, np.array(ret))

                self._train_episode += 1
                if self.cfg.wandb_log:
                    plt.plot(xs, ret, xs, values)
                    wandb.log({"ret_val_diff" : plt})
                    episode_metrics = dict()
                    episode_metrics['episodic_length'] = info["episode"]["l"][0]
                    episode_metrics['episodic_return'] = info["episode"]["r"][0]
                    episode_metrics['steps_per_second'] = info["episode"]["l"][0]/(time.time() - episode_start_time)
                    episode_metrics['env_buffer_length'] = len(self.agent.env_buffer)
                    wandb.log(episode_metrics, step=self._train_step)
                rew = []
                val_seq = []
                state_seq = []
                action_seq = []

                state, info = self.train_env.reset(seed=self.cfg.seed)
                initial_state = state
                done = False
                episode_start_time = time.time()
            else:
                state = next_state

        self.train_env.close()
    
    def _eval(self):
        returns = 0 
        steps = 0
        for _ in range(self.cfg.num_eval_episodes):
            done = False 
            state, info = self.eval_env.reset(seed=self.cfg.seed + 100)
            while not done:
                action = self.agent.get_action(state, self._train_step, True)
                next_state, _, done, trunc, info = self.eval_env.step(action)
                done = done or trunc
                state = next_state
                
            returns += info["episode"]["r"]
            steps += info["episode"]["l"]
            
            print("Episode: {}, total numsteps: {}, return: {}".format(self._train_episode, self._train_step, round(info["episode"]["r"], 2)))

        eval_metrics = dict()
        eval_metrics['eval_episodic_return'] = returns/self.cfg.num_eval_episodes
        eval_metrics['eval_episodic_length'] = steps/self.cfg.num_eval_episodes

        if self.cfg.save_snapshot and returns/self.cfg.num_eval_episodes >= self._best_eval_returns:
            self.save_snapshot(best=True)
            self._best_eval_returns = returns/self.cfg.num_eval_episodes

        if self.cfg.wandb_log:
            wandb.log(eval_metrics, step = self._train_step)

    def _render_episodes(self, record):
        frames = []
        done = False 
        state, info = self.eval_env.reset(seed=self.cfg.seed)
        while not done:
            action = self.agent.get_action(state, self._train_step, True)
            next_state, _, done, trunc, info = self.eval_env.step(action)
            done = done or trunc
            self.eval_env.render()
            state = next_state
        if record:
            save_frames_as_gif(frames)
        print("Episode: {}, episode steps: {}, episode returns: {}".format(i, info["episode"]["l"], round(info["episode"]["r"], 2)))
        
    def _eval_bias(self): # not used?
        final_mc_list, final_obs_list, final_act_list = self._mc_returns()
        final_mc_norm_list = np.abs(final_mc_list.copy())
        final_mc_norm_list[final_mc_norm_list < 10] = 10

        obs_tensor = torch.FloatTensor(final_obs_list).to(self.device)
        acts_tensor = torch.FloatTensor(final_act_list).to(self.device)
        lower_bound = self.agent.get_lower_bound(obs_tensor, acts_tensor)
        
        bias = final_mc_list - lower_bound
        normalized_bias_per_state = bias / final_mc_norm_list

        if self.cfg.wandb_log:
            metrics = dict()
            metrics['mean_bias'] = np.mean(bias)
            metrics['std_bias'] = np.std(bias)
            metrics['mean_normalised_bias'] = np.mean(normalized_bias_per_state)
            metrics['std_normalised_bias'] = np.std(normalized_bias_per_state)
            wandb.log(metrics, step = self._train_step)

    def _mc_returns(self):
        final_mc_list = np.zeros(0)
        final_obs_list = []
        final_act_list = [] 
        n_mc_eval = 1000
        n_mc_cutoff = 350

        while final_mc_list.shape[0] < n_mc_eval:
            o, i = self.eval_env.reset(seed=self.cfg.seed)
            reward_list, obs_list, act_list = [], [], []
            r, d, ep_ret, ep_len = 0, False, 0, 0

            while not d:
                a = self.agent.get_action(o, self._train_step, True)
                obs_list.append(o)
                act_list.append(a)
                o, r, d, t, _ = self.eval_env.step(a)
                d = d or t
                ep_ret += r
                ep_len += 1
                reward_list.append(r)

            discounted_return_list = np.zeros(ep_len)
            for i_step in range(ep_len - 1, -1, -1):
                if i_step == ep_len -1 :
                    discounted_return_list[i_step] = reward_list[i_step]
                else :
                    discounted_return_list[i_step] = reward_list[i_step] + self.cfg.gamma * discounted_return_list[i_step + 1]

            final_mc_list = np.concatenate((final_mc_list, discounted_return_list[:n_mc_cutoff]))
            final_obs_list += obs_list[:n_mc_cutoff]
            final_act_list += act_list[:n_mc_cutoff]

        return final_mc_list, np.array(final_obs_list), np.array(final_act_list)

    def save_snapshot(self, best=False):
        if best:
            snapshot = Path(self.checkpoint_path) / 'best.pt'
        else:
            snapshot = Path(self.checkpoint_path) / Path(str(self._train_step)+'.pt')
        save_dict = self.agent.get_save_dict()
        torch.save(save_dict, snapshot)
