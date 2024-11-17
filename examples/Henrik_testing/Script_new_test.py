import sys
import os
import datetime
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from config.utils import load_config
from external.utils import load_data
from envs.env import VerticalFarmEnv

# Define simulation parameters
start_date = '2023-04-01'
end_date = '2023-04-21'  # Define the end date
steps_per_day = 24       # Number of steps per day (e.g., hourly steps)

start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')

# Load configuration and weather data
config = load_config('config/')
data = load_data('external/weather/data', start_datetime, end_datetime)

# Initialize environment
env = VerticalFarmEnv(
    start_datetime=start_datetime,
    config=config,
    data=data,
    end_datetime=end_datetime,
    steps_per_day=steps_per_day,
    render_mode='file'
)



# Reset and run the simulation
obs, _ = env.reset()
i = 0

# while True:
#     action = np.array([1, 1, 0, 0, 0, 0, 1])  # Example action
#     obs, rewards, terminated, truncated, info = env.step(action)
#     env.render()

#     if terminated:
#         env.close()
#         break

#     i += 1

for step in range(473):
    action = np.array([1, 1, 0, 0, 0, 0, 1])  # Example action
    obs, rewards, terminated, truncated, info = env.step(action)
    env.render()

env.close()
