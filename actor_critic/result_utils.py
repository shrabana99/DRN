import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def plot_result(args):
    window = 100  
    plt.figure(figsize=(12, 6)) 

    log_file_path = args.log_dir + '/' +  args.env_name  
    data = pd.read_csv(log_file_path)
    y = data['reward_mean'].values
    x = np.arange(1, len(y) + 1)
    # z = pd.Series(y).rolling(window).mean()

    plt.plot(x, y, color='r', label='update')
    # plt.plot(x, z, color='g', label='average')

    plt.xlabel("episodes")
    plt.ylabel("true/average rewads")
    plt.title("episode vs reward, (moving average of {} - episode)".format(window))

    plt.legend()
    plt.savefig(args.env_name)

