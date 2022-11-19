"""
NNI hyperparameter optimization example.

Check the online tutorial for details:
https://nni.readthedocs.io/en/stable/tutorials/hpo_quickstart_pytorch/main.html
"""

from pathlib import Path
import signal

from nni.experiment import Experiment

# Define search space
search_space = {
    'w_1': {'_type': 'uniform', '_value': [0.0, 1.0]},
    'w_2': {'_type': 'uniform', '_value': [0.0, 1.0]},
    'w_3': {'_type': 'uniform', '_value': [0.0, 1.0]},
    'w_4': {'_type': 'uniform', '_value': [0.0, 1.0]},
    'w_5': {'_type': 'uniform', '_value': [0.0, 1.0]},
    'w_6': {'_type': 'uniform', '_value': [0.0, 1.0]},
    'dropout_rate': {'_type': 'uniform', '_value': [0.1, 0.9]},
    'learning_rate': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
}
# print(Path(__file__).parent)
# Configure experiment
experiment = Experiment('local')
experiment.config.trial_command = 'python3 model.py'
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.search_space = search_space
experiment.config.tuner.name = 'GP'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 1

# Run it!
experiment.run(port=8080, wait_completion=False)

print('Experiment is running. Press Ctrl-C to quit.')
signal.pause()
