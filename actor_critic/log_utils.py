import pandas as pd
import os

class Log: 
    def __init__(self, args):
        self.args = args
        # save training log
        if not os.path.exists(self.args.log_dir):
            os.mkdir(self.args.log_dir)
        self.log_file_path = self.args.log_dir + '/' +  self.args.env_name  
        self.log = {'time_stamp': [], 'update': [], \
                    'actor_loss': [], 'critic_loss': [], 'entropy': [], \
                    'reward_min': [], 'reward_mean': [], 'reward_max': [] }


    def append_single_log(self, single_log): # t_s, update, al, vl, ent, r_min, r_mean, r_max):
        for key, val in zip(self.log.keys(), single_log): 
            self.log[key].append(val)


    def save_log(self):  
        data = pd.DataFrame.from_dict(self.log)
        data.to_csv( self.log_file_path, sep = ',', mode = 'w')
                    
