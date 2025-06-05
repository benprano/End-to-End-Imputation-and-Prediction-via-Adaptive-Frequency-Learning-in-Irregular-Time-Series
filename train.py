import os
import random
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.DyFAIP_Network import DyFAIP_Aware
from utils.afail_loss import Afail
from utils.missing_mecanisms import DataSampler
from utils.train_evaluate_helpers import TrainerHelpers


def seed_all(seed: int = 1992):
    """Seed all random number generators."""
    print("Using Seed Number {}".format(seed))

    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    # set fixed value for python built-in pseudo-random generator
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
seed_all()


dn="-------- path to the dataset"
task_dataset ="BEIJINGAIRQUALITY_24_HRS_DATA_128"

all_dataset_loader = np.load(os.path.join(os.path.join(dn,task_dataset), "train_test_data.npz"),allow_pickle=True)
train_val_loader = all_dataset_loader['folds_data_train_valid']
test_loader = all_dataset_loader['folds_data_test']
dataset_settings = np.load(os.path.join(os.path.join(dn,task_dataset), "data_max_min.npz"), allow_pickle=True)

data_max, data_min= dataset_settings['data_max'], dataset_settings['data_min']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_length = dataset_settings['seq_length'].item()
input_dim = dataset_settings['input_dim'].item()
hidden_dim, output_dim  = 128, 1#dataset_settings['output_size'].item()

LEARNING_RATE = 1e-3
optimizer_config={"lr": 1e-3, "betas": (0.9, 0.98), "eps": 4e-09, "weight_decay": 5e-4}
NUM_EPOCHS =500
NUM_FOLDS = 10
model_name="DyFAIP".lower()
n_patience = 100
batch_size=64
steps_per_epoch = int(dataset_settings['shape_data'][0] / batch_size / NUM_FOLDS)
total_steps_per_fold = int(steps_per_epoch * NUM_EPOCHS)
num_warmup_steps = int(0.1 * total_steps_per_fold)

arr = np.arange(input_dim)
percen = 0.2
num_samples = int(len(arr) * percen)
sampled = np.random.choice(arr, size=num_samples, replace=False)
# INFO[LOADING MODEL]
print("INFO[LOADING MODEL]")

dyfaip_aware = DyFAIP_Aware(input_dim, hidden_dim, seq_length, output_dim).to(device)
# Create an instance of the class
data_sampler =DataSampler(percentage=0.2, mode='MCAR')

loss_calculator = Afail()
optimizer = torch.optim.Adam(dyfaip_aware.parameters(), **optimizer_config)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=NUM_EPOCHS,
                                          steps_per_epoch=100)
criterion = nn.MSELoss().to(device)
best_model_wts = deepcopy(dyfaip_aware.state_dict())

print(dyfaip_aware)

train_valid_inference = TrainerHelpers(input_dim, hidden_dim, seq_length, output_dim,
                                       device, optimizer, criterion,loss_calculator, scheduler,
                                       data_sampler,NUM_EPOCHS, patience_n=n_patience, task=False)

main_path = f"path to save the results for both tasks"
task_path=f"{os.path.join(main_path, task_dataset.split('_')[0])}"
if not os.path.exists(task_path):
    os.makedirs(task_path)

scores_folds= []
for idx, (train_loader, test_data) in enumerate(zip(train_val_loader ,test_loader)):
    print(f'[INFO]: Training on fold : {idx+1}')
    # Reset the model weights
    dyfa_aware.load_state_dict(best_model_wts)
    train_data, valid_data= train_loader
    scores_= train_valid_inference.train_validate_evaluate(DyFAIP_Aware, dyfaip_aware, idx+1,train_data,
                                                           valid_data, test_data, dataset_settings,
                                                           task_path)
    scores_folds.append(scores_)


mse_mae_r2_task =np.mean([fold[0][0] for fold in scores_folds], axis=0)
mse_mae_r2_task_imp = np.mean([fold[0][1] for fold in scores_folds], axis=0)

np.savez(os.path.join(task_path, f"test_data_fold_{task_dataset.split('_')[0]}_results.npz".lower()),
                      reg_scores=scores_folds ,
                      task_results=mse_mae_r2_task,
                      imputation_results= mse_mae_r2_task_imp)

print("mse_mae_r2_task, mse_mae_r2_task_imp", mse_mae_r2_task, mse_mae_r2_task_imp)
