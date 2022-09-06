import numpy as np
import torch
from models import ActorCritic
from datetime import datetime
from utils import discount_with_dones
import os
from result_utils import plot_result

class a2c_agent:
    def __init__(self, envs, action_space, log, args):
        self.envs = envs
        self.args = args
        self.action_space = action_space
        self.log = log
        
        self.num_batch = args.num_workers * args.nsteps
        self.batch_shape = (self.num_batch, 4, 84, 84)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # define the network
        self.model = ActorCritic(action_space.n, self.device).to(self.device)

        # define the optimizer
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.args.lr, eps=self.args.eps, alpha=self.args.alpha)
    
        # save model parameters
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.model_path = self.args.save_dir + self.args.env_name + '/'
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        
                    
        # get the obs..
        self.obs = self.envs.reset()
        print("2........................", self.obs.shape)




    # train the network..
    def learn(self):
        # get the reward to calculate other information
        episode_rewards = np.zeros((self.args.num_workers, ), dtype=np.float32)
        final_rewards = np.zeros((self.args.num_workers, ), dtype=np.float32)
        num_updates = self.args.total_frames // self.num_batch 

        # start to update
        for update in range(1, num_updates+1):
            ######### collect nstep #########
            n_states, n_rewards, n_actions, n_dones = [], [], [], []
            
            for n in range(self.args.nsteps): 
                # take action  and store
                n_states.append(np.copy(self.obs))
                action_col = self.model.get_action(self.obs) 
                action = action_col.squeeze(1)
                n_actions.append(action)
                self.obs, reward, done, _ = self.envs.step(action)
                n_rewards.append(reward)
                n_dones.append(done)

                episode_rewards += reward
                # get the masks
                masks = 1 - done # np.array([0.0 if done else 1.0 for done in dones], dtype=np.float32)
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks
                # print(action, final_rewards, episode_rewards)

            ######### process the rollouts #########
            n_states = np.asarray(n_states, dtype=np.uint8).swapaxes(1, 0).reshape( self.batch_shape )
            n_actions = np.asarray(n_actions, dtype=np.int32).swapaxes(1, 0) # (num_actions, nsteps)
            n_rewards = np.asarray(n_rewards, dtype=np.float32).swapaxes(1, 0)
            n_dones = np.asarray(n_dones, dtype=bool).swapaxes(1, 0) 

            # compute returns
            last_values = self.model.get_value(self.obs).squeeze(1)
            for n, (rewards, dones, value) in enumerate(zip(n_rewards, n_dones, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                gamma = self.args.gamma
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards+[value], dones+[0], gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, gamma)
                n_rewards[n] = rewards

            # start to update network
            vl, al, ent = self._update_network(n_states, n_rewards, n_actions)
            if update % self.args.log_interval == 0:
                self.log.append_single_log( [datetime.now(), update, al, vl, ent, final_rewards.min(), final_rewards.mean(), final_rewards.max()] )
                print( '[{}] Update: {}/{}, Frames: {}, Rewards: {:.1f}, VL: {:.3f}, PL: {:.3f}, Ent: {:.2f}, Min: {}, Max:{}'.format(\
                    datetime.now(), update, num_updates, (update+1)*(self.args.num_workers * self.args.nsteps),\
                    final_rewards.mean(), vl, al, ent, final_rewards.min(), final_rewards.max()) )
                torch.save(self.model.state_dict(), self.model_path + 'model.pt')

            if update % (self.args.log_interval * 10) == 0:
                self.log.save_log()
                plot_result(self.args)



    # update_network
    def _update_network(self, n_states, n_rewards, n_actions):
        n_states = self.model.get_tensor(n_states)
        values, pi = self.model(n_states)  # torch.Size([nbatch, 1]) torch.Size([nbacth, n_action])

        returns = self.model.get_tensor(n_rewards.flatten()).unsqueeze(1) # torch.Size([nbatch, 1])
        actions = self.model.get_tensor(n_actions.flatten(), dtype=torch.int64).unsqueeze(1)

        action_log_probs, dist_entropy = self.model.evaluate_actions(pi, actions)
        # print( action_log_probs.size() )
        advantages = returns - values

        # get the value loss
        value_loss = advantages.pow(2).mean()
        # get the action loss
        action_loss = -(advantages.detach() * action_log_probs).mean()
        # total loss
        total_loss = action_loss + self.args.value_loss_coef * value_loss - self.args.entropy_coef * dist_entropy
        # start to update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        return value_loss.item(), action_loss.item(), dist_entropy.item()

