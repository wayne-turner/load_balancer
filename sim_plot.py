import matplotlib.pyplot as plt
import imageio
import numpy as np
from scipy.interpolate import interp1d, make_interp_spline

def smooth_line(x, y, num_points=300):
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]

    if len(np.unique(x)) > 3: 
        x_new = np.linspace(x.min(), x.max(), num_points)
        try:
            spl = make_interp_spline(x, y, k=3) 
            y_smooth = spl(x_new)
        except ValueError:
            lin_interp = interp1d(x, y, kind='linear')
            y_smooth = lin_interp(x_new)
    else:
        x_new = np.linspace(x.min(), x.max(), num_points)
        lin_interp = interp1d(x, y, kind='linear')
        y_smooth = lin_interp(x_new)

    return x_new, y_smooth

all_values = np.concatenate([env.demand_history, env.storage_history, env.rewards_history])
y_min, y_max = all_values.min(), all_values.max()
y_limit_buffer = (y_max - y_min) * 0.05 

frames = [] 

for i in range(1, len(env.time_steps_history) + 1):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_facecolor('#121212') 
    plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
    
    if i > 1: 
        x_new, demand_smooth = smooth_line(np.array(env.time_steps_history[:i]), np.array(env.demand_history[:i]))
        x_new, storage_smooth = smooth_line(np.array(env.time_steps_history[:i]), np.array(env.storage_history[:i]))
        x_new, rewards_smooth = smooth_line(np.array(env.time_steps_history[:i]), np.array(env.rewards_history[:i]))
        
        plt.plot(x_new, demand_smooth, label='Demand', color='cyan')
        plt.plot(x_new, storage_smooth, label='Storage', color='magenta')
        plt.plot(x_new, rewards_smooth, label='Rewards', color='yellow')
    else:
        plt.plot(env.time_steps_history[:i], env.demand_history[:i], label='Demand', color='cyan')
        plt.plot(env.time_steps_history[:i], env.storage_history[:i], label='Storage', color='magenta')
        plt.plot(env.time_steps_history[:i], env.rewards_history[:i], label='Rewards', color='yellow')
    
    plt.title('Smart Grid Simulation Results', color='white')
    plt.xlabel('Time Step', color='white')
    plt.ylabel('Value', color='white')
    plt.legend(title='Metric')
    plt.xlim([env.time_steps_history[0], env.time_steps_history[-1]])
    plt.ylim([y_min - y_limit_buffer, y_max + y_limit_buffer]) 
    
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    frame_filename = f'frame_{i:03d}.png'
    plt.savefig(frame_filename, facecolor='#121212')
    frames.append(frame_filename)
    plt.close()


repeat_last_frame = 30
standard_duration = 0.1

with imageio.get_writer('sim_results.gif', mode='I', duration=standard_duration) as writer:
    for frame_filename in frames:
        image = imageio.imread(frame_filename)
        writer.append_data(image)

    last_image = imageio.imread(frames[-1])
    for _ in range(repeat_last_frame):
        writer.append_data(last_image) 
