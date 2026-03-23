#%%

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
from CNN_CVAE import CVAE_encoder  # Use own CNN_CVAE.py file
import New_Minkowski_Dataset as md # Use own New_Minkowski_Dataset.py file to load specific data
from skimage.measure import euler_number
from skimage.measure import perimeter_crofton
from scipy.stats    import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
import json
import optuna
from skimage import filters
from scipy.ndimage import gaussian_filter
import pandas as pd, seaborn as sns
import random
SEED = 20
SCRATCH = os.environ.get("SCRATCH", ".")
OUT_DIR = os.path.join(SCRATCH, f"BO_midterm_optuna_{SEED}")
os.makedirs(OUT_DIR, exist_ok=True)

def set_global_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)         
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_global_seed(0) 
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


parent_dir = os.path.join("DAT_files", "40")
folders = [os.path.join(parent_dir, folder) for folder in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, folder))]
dataset_40 = md.MultiFolderMinkowskiDataset6(folders)

test_dir = os.path.join("DAT_files", "10")
folders_test = [os.path.join(test_dir, folder) for folder in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, folder))]
dataset_test = md.MultiFolderMinkowskiDataset6(folders_test)

whole_dir = "DAT_files_ori"
folders_whole = [os.path.join(whole_dir, folder) for folder in os.listdir(whole_dir) if os.path.isdir(os.path.join(whole_dir, folder))]
dataset_whole = md.MultiFolderMinkowskiDataset6(folders_whole)

# Split into training and validation datasets
total_size = len(dataset_40)
train_size = int(0.8 * total_size)
val_size = int(0.2 * total_size)
train_dataset, val_dataset = random_split(dataset_40, [train_size, val_size], generator=torch.Generator().manual_seed(0))

#%%
global_mfs_40 = [dataset_40[i][2] for i in range(0, len(dataset_40), 400)]
# global_cond_trainval = torch.stack(global_mfs_40)  # shape: [40, 3]
# global_sigma_40 = [dataset_40[i][3] for i in range(0, len(dataset_40), 400)]
# global_sigma_trainval = torch.stack(global_sigma_40)  # shape: [40, 3]

# global_mfs_10 = [dataset_test[i][2] for i in range(0, len(dataset_test), 400)]
# global_mfs_test = torch.stack(global_mfs_10)  # shape: [10, 3]
# global_sigma_10 = [dataset_test[i][3] for i in range(0, len(dataset_test), 400)]
# global_sigma_test = torch.stack(global_sigma_10)  # shape: [10, 3]

global_mfs_50 = [dataset_whole[i][2] for i in range(0, len(dataset_whole), 400)]
global_mfs_whole = torch.stack(global_mfs_50)  
global_sigma_50 = [dataset_whole[i][3] for i in range(0, len(dataset_whole), 400)]
global_sigma_whole = torch.stack(global_sigma_50)


#%%

BIN_CFG = dict(sigma=0.2)  
def binarize(img, sigma=0.8):
    smooth = gaussian_filter(img, sigma=sigma)
    thresh = filters.threshold_otsu(smooth)
    bin_img = smooth >= thresh
    return bin_img.astype(bool)

def calculate_minkowski_functionals(binary_image):
    v0 = np.sum(binary_image) / binary_image.size
    # v1 = perimeter(binary_image, 4) / binary_image.shape[0]
    v1 = perimeter_crofton(binary_image, 4) / binary_image.shape[0] 
    im_inv = np.logical_not(binary_image) 
    v2 = euler_number(im_inv, connectivity=1)
    return np.array([v0, v1, v2])

# Sample 10 validation samples for visualization
def sample_random_val_images(val_dataset, num_samples=5):
    idxs = np.random.choice(len(val_dataset), num_samples, replace=False)
    samples = []
    for i in idxs:
        img, local_mfs, global_mfs, _ = val_dataset[i]  
        samples.append((img, local_mfs, global_mfs))      
    return samples
fixed_val_samples = sample_random_val_images(val_dataset, num_samples=5)

def visualize_fixed_reconstructions(model, fixed_samples, epoch, device, save_dir="recon_history"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    n = len(fixed_samples)
    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
    local_list, mf_list = [], []

    for i, (x, local_mfs, global_mfs) in enumerate(fixed_samples):
        # ------------- forward -------------
        x_tensor      = x.unsqueeze(0).to(device)          # [1,1,H,W]
        cond_tensor   = global_mfs.unsqueeze(0).to(device) # [1,3]
        with torch.no_grad():
            recon_logits, _, _ = model(x_tensor, cond_tensor)
            recon = torch.sigmoid(recon_logits).squeeze().cpu().numpy()  # [H,W]

        bin_img = binarize(recon, **BIN_CFG)
        mf_vals = calculate_minkowski_functionals(bin_img)         # (3,)
        mf_list.append(mf_vals)

        local_vals = local_mfs.cpu().numpy()                       # (3,)
        local_list.append(local_vals)

        local_str = ",".join(f"{v:.3f}" for v in local_vals)
        mf_str    = ",".join(f"{v:.3f}" for v in mf_vals)


        axes[0, i].imshow(x.cpu().numpy().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Original\nLocal: [{local_str}]", fontsize=10)

        axes[1, i].imshow(recon, cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title(f"Epoch {epoch}", fontsize=10)

        axes[2, i].imshow(bin_img, cmap="gray")
        axes[2, i].axis("off")
        axes[2, i].set_title(f"Thresh img\nMF: [{mf_str}]", fontsize=10)

    local_arr = np.stack(local_list)   # (N,3)
    mf_arr    = np.stack(mf_list)      # (N,3)
    stats_txt = []
    r2_all = []
    for j, name in enumerate(["Porosity", "surface area", "Euler characteristic"]):
        rho, _ = pearsonr(local_arr[:, j], mf_arr[:, j])
        r2     = r2_score(local_arr[:, j], mf_arr[:, j])
        r2_all.append(r2)
        stats_txt.append(f"{name}: ρ={rho:.3f}, R²={r2:.3f}")

    fig.text(0.5, -0.03, " | ".join(stats_txt),
             ha="center", va="top", fontsize=8)

    fig.tight_layout()
    save_path = os.path.join(save_dir, f"epoch_{epoch:03d}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return r2_all  

def train_model(model, train_loader, val_loader, optimizer, num_epochs, device, save_dir, fixed_val_samples,beta):
    train_losses = []
    val_losses = []
    train_rec_epoch, train_kld_epoch = [], []
    val_rec_epoch,   val_kld_epoch   = [], []
    epoch_metrics = []
    early_stopping_patience = 20
    early_stopping_counter = 0
    best_val_loss = float('inf')
    # criterion = nn.MSELoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    os.makedirs(save_dir, exist_ok=True)
    best_epoch    = 0       


    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        rec_sum = kld_sum = 0
        beta_t = beta

        for x, _, global_mfs, _ in train_loader:
            x = x.to(device)
            cond = global_mfs.to(device)
            recon_image, mu, log_var = model(x, cond)
            kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            recon_loss = criterion(recon_image, x)
            total_loss = recon_loss + beta_t*kld_loss  

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_train_loss += total_loss.item()
            rec_sum += recon_loss.item()
            kld_sum += kld_loss.item()
        train_rec_epoch.append(rec_sum / len(train_loader))
        train_kld_epoch.append(kld_sum / len(train_loader))

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        rec_sum_val = kld_sum_val = 0
        with torch.no_grad():
            for x, _, global_mfs, _ in val_loader:
                x = x.to(device)
                cond = global_mfs.to(device)
                recon_image, mu, log_var = model(x, cond)
                kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                recon_loss = criterion(recon_image, x)
                total_loss = recon_loss + beta_t*kld_loss

                running_val_loss += total_loss.item()
                rec_sum_val += recon_loss.item()
                kld_sum_val += kld_loss.item()
            val_rec_epoch.append(rec_sum_val / len(val_loader))
            val_kld_epoch.append(kld_sum_val / len(val_loader))
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")            

        epoch_metrics.append({
            "epoch"        : epoch,
            "beta_now"     : beta_t,
            "train_loss"   : avg_train_loss,
            "val_loss"     : avg_val_loss,
            "train_rec"    : train_rec_epoch[-1],
            "train_kld"    : train_kld_epoch[-1],
            "val_rec"      : val_rec_epoch[-1],
            "val_kld"      : val_kld_epoch[-1]
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch    = epoch + 1      
            early_stopping_counter = 0

            torch.save(model.state_dict(),
                       os.path.join(save_dir, "model_best.pth"))

            torch.save({
                "epoch" : epoch,  
                "model_state_dict" : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
                "val_loss" : best_val_loss,
            }, os.path.join(save_dir, "checkpoint_best.pth"))

        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    torch.save(model.state_dict(),
               os.path.join(save_dir, "model_final.pth"))

    torch.save({
        "epoch" : epoch+1 ,
        "best_epoch"   : best_epoch, 
        "model_state_dict" : model.state_dict(),
        "optimizer_state_dict" : optimizer.state_dict(),
        "val_loss" : avg_val_loss,
    }, os.path.join(save_dir, "checkpoint_last.pth"))

    final_r2 = visualize_fixed_reconstructions(model, fixed_val_samples, epoch=best_epoch, device=device, save_dir=save_dir)

    # Save final loss plot
    plt.figure()
    plt.plot(train_rec_epoch, label="Train Recon Loss")
    plt.plot(val_rec_epoch, label="Val Recon Loss")
    plt.plot(train_kld_epoch, label="Train KLD Loss")
    plt.plot(val_kld_epoch, label="Val KLD Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Reconstruction and KLD Loss")
    plt.savefig(os.path.join(save_dir, "loss_components.png"))
    plt.close()

    return train_losses, val_losses, epoch_metrics, final_r2, best_epoch


def generate_random_images(model,
                           num_samples,
                           latent_shape,
                           device,
                           cond,
                           sigma,
                           save_dir="generated_imgs",
                           parity_plot=True,
                           metrics_csv_path=None):

    set_global_seed(0) 
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    cond = torch.as_tensor(cond, dtype=torch.float32)
    if cond.dim() == 1:
        cond = cond.unsqueeze(0)           # [1,3]
    N = cond.size(0)
    sigma = torch.as_tensor(sigma, dtype=torch.float32)
    if sigma.dim() == 1:
        sigma = sigma.unsqueeze(0)         # [1,1]

    C, H, W = latent_shape
    mu_list, std_list,mse_each_list = [], [], []
    for i in range(N):
        c = cond[i].to(device)             # [3]
        sigma_i = sigma[i].to(device)      
        with torch.no_grad():
            z = torch.randn(num_samples, C, H, W, device=device)
            cond_map = c.unsqueeze(0).repeat(num_samples, 1) \
                     .unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            recon = torch.sigmoid(model.decoder(torch.cat([z, cond_map], 1))).cpu()

        mf_vals = []
        for img in recon:
            bin_img = binarize(img[0].numpy(), **BIN_CFG)
            mf_vals.append(calculate_minkowski_functionals(bin_img))
        mf_vals = np.stack(mf_vals)              # [100,3]

      
        mu = mf_vals.mean(0) 
        mu_list.append(mu)
        std_this = mf_vals.std(0) 
        std_list.append(std_this)
        c_np  = c.cpu().numpy().reshape(1, -1)   # (1, 3)
        mu_np = mu.reshape(1, -1) 
        mse_this = mean_squared_error(c_np, mu_np, multioutput='raw_values') 
        mse_each_list.append(mse_this)  

      
        ncols = int(num_samples**0.5)
        nrows = (num_samples + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(1.8*ncols, 1.8*nrows),
                                 squeeze=False)
        idx = 0
        for r in range(nrows):
            for s in range(ncols):
                axes[r, s].axis("off")
                if idx < num_samples:
                    axes[r, s].imshow(recon[idx].squeeze(), cmap="gray")
                    idx += 1
        names = ["Porosity", "surface area", "Euler characteristic"]            
        cond_str = ", ".join(f"{n}={c[j]:.3f}"      for j, n in enumerate(names))
        sigma_str = ", ".join(f"{n}={sigma_i[j]:.3f}"      for j, n in enumerate(names))
        mu_str   = ", ".join(f"{n}={mu[j]:.3f}"     for j, n in enumerate(names))
        std_str  = ", ".join(f"{n}={std_this[j]:.3f}"for j, n in enumerate(names))
        mse_str = ", ".join(f"{n}={mse_this[j]:.3f}" for j, n in enumerate(names))


        fig.suptitle(
            f"{cond_str} | global σ: {sigma_str}\n"           
            f"μ: {mu_str} | σ: {std_str} | MSE: {mse_str}",   
            fontsize=20, y=1.02
        )
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"grid_{i:02d}.png"),
                    dpi=120, bbox_inches="tight")
        plt.close(fig)


    # ---------- parity plot ----------
    mu_all   = np.stack(mu_list)           # [N,3]
    std_all = np.stack(std_list)         # [N,3]
    cond_all = cond.cpu().numpy()          # [N,3]
    sigma_all = sigma.cpu().numpy()        # [N,3]
    r2_overall = [r2_score(cond_all[:, j], mu_all[:, j]) for j in range(3)] 
    r2_std = [r2_score(sigma_all[:, j], std_all[:, j]) for j in range(3)]
    mse_each_cond = np.stack(mse_each_list)  # [N,3]


    eps = 1e-8
    cv_target = sigma_all / (cond_all + eps)
    cv_pred   = std_all   / (mu_all   + eps)

    r2_cv = [r2_score(cv_target[:, j], cv_pred[:, j]) for j in range(3)]

    if parity_plot:
        names = ["Porosity", "surface area", "Euler characteristic"]

        # ========= 1) μ parity (cond vs μ_pred) =========
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=False, sharey=False)

        for j, ax in enumerate(axes):
            ax.scatter(cond_all[:, j], mu_all[:, j], s=12)      
            lims = [min(cond_all[:, j].min(), mu_all[:, j].min()),
                    max(cond_all[:, j].max(), mu_all[:, j].max())]
            ax.plot(lims, lims, "--", linewidth=1)              # y = x
            ax.set_xlabel(f"Target {names[j]}")
            ax.set_ylabel(f"Pred μ {names[j]}")
            ax.set_title(f"{names[j]}  R²={r2_overall[j]:.3f}")

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "parity_mu_all.png"), dpi=150)
        plt.close(fig)

        # ========= 2) σ parity normalised (sigma_target vs sigma_pred) =========
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        for j, ax in enumerate(axes):
            ax.scatter(cv_target[:, j], cv_pred[:, j], s=12)
            lims = [min(cv_target[:, j].min(), cv_pred[:, j].min()),
                    max(cv_target[:, j].max(), cv_pred[:, j].max())]
            ax.plot(lims, lims, "--", lw=1)                 # y = x
            ax.set_xlabel(f"Target CV {names[j]}")
            ax.set_ylabel(f"Pred CV {names[j]}")
            ax.set_title(f"{names[j]}  R²={r2_cv[j]:.3f}")

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "parity_sigma_cv.png"), dpi=150)
        plt.close(fig)


    if metrics_csv_path is None:                     
        metrics_csv_path = os.path.join(save_dir, "metrics.csv")

    if metrics_csv_path:                              
        df = pd.DataFrame({
            "global_mfs_idx": np.arange(N),
            "target_V0": cond_all[:, 0],"target_std_V0": sigma_all[:, 0],"pred_V0": mu_all[:, 0],"std_V0": std_all[:, 0],
            "target_V1": cond_all[:, 1],"target_std_V1": sigma_all[:, 1],"pred_V1": mu_all[:, 1],"std_V1": std_all[:, 1],
            "target_V2": cond_all[:, 2],"target_std_V2": sigma_all[:, 2],"pred_V2": mu_all[:, 2],"std_V2": std_all[:, 2],
            "MSE_V0": mse_each_cond[:, 0],
            "MSE_V1": mse_each_cond[:, 1],
            "MSE_V2": mse_each_cond[:, 2],
        })
        os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)
        df.to_csv(metrics_csv_path, index=False)
        print(f"metrics saved to {metrics_csv_path}")

        violin_plots(metrics_csv_path, save_dir)
    return r2_overall, r2_cv, mse_each_cond, mu_all

def violin_plots(metrics_csv_path, save_dir):

    df = pd.read_csv(metrics_csv_path)

    mf_names = ["V0", "V1", "V2"]
    for mf in mf_names:
        data = [
            df[f"target_{mf}"].values,   
            df[f"pred_{mf}"].values      
        ]

        plt.figure(figsize=(4,4))
        sns.violinplot(data=data, inner="box")   # inner='box' 
        plt.xticks([0,1], ["Target", "Generated"], fontsize=12, fontstyle="italic")
        plt.ylabel(mf)
        plt.title(f"{mf}: Target vs Generated", fontsize=14)
        plt.tight_layout()

        out_path = os.path.join(save_dir, f"violin_{mf}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

def build_model(latent, ks):
    return CVAE_encoder(
        channel_in=channel_in,
        ch=ch,
        blocks=blocks,
        latent_channels=latent,
        num_res_blocks=num_res_blocks,
        norm_type=norm_type,
        deep_model=deep_model,
        condition_dim=condition_dim,
        kernel_size=ks
    ).to(device)

def objective(trial):
    set_global_seed(0)
    g = torch.Generator().manual_seed(0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    ks     = 3
    latent = trial.suggest_int ("latent", 10, 35)        
    beta   = trial.suggest_float("beta",  0.0, 1.0)      

    run_name  = f"b{beta:.3f}_l{latent}"
    save_root = os.path.join(OUT_DIR, "runs", run_name)
    os.makedirs(save_root, exist_ok=True)

    model     = build_model(latent, ks)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    _, _, _, _, best_epoch = train_model(
        model, train_loader, val_loader, optimizer,
        num_epochs=num_epochs, device=device,
        save_dir=os.path.join(save_root, "recon_history"),
        fixed_val_samples=fixed_val_samples,
        beta=beta)

    latent_shape = (latent, 5, 5)    
    r2_overall, r2_cv,  _, _ = generate_random_images(
        model,
        latent_shape=latent_shape,
        device=device,
        num_samples=400,
        cond=global_mfs_whole,              
        sigma=global_sigma_whole,          
        save_dir=os.path.join(save_root, "generated_imgs"),
        parity_plot=True,
        metrics_csv_path=os.path.join(save_root, "generated_imgs", "metrics.csv")
    )

    trial.set_user_attr("r2_V0", r2_overall[0])
    trial.set_user_attr("r2_V1", r2_overall[1])
    trial.set_user_attr("r2_V2", r2_overall[2])
    trial.set_user_attr("r2_std_V0", r2_cv[0])
    trial.set_user_attr("r2_std_V1", r2_cv[1])
    trial.set_user_attr("r2_std_V2", r2_cv[2])
    trial.set_user_attr("best_epoch", best_epoch)

    del model; torch.cuda.empty_cache()
    return  0.95 * np.mean(r2_overall) + 0.05 * np.mean(r2_cv)       

# Define model parameters
channel_in = 1  # Grayscale images
condition_dim = 3   
ch = 16  # Base number of channels
blocks = (1, 2, 4)  # Adjust based on your architecture
# blocks = (1, 2)
num_res_blocks = 1
norm_type = "bn"  # Batch normalization
deep_model = False  # Adjust based on your preference

# Grid Search
batch_size = 40
num_epochs = 150


if __name__ == "__main__":
    storage_url = f"sqlite:///{OUT_DIR}/BO_midterm_optuna_{SEED}.db"

    study = optuna.create_study(
        study_name=f"BO_midterm_optuna_{SEED}",
        direction="maximize",
        storage=storage_url,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=SEED,
            n_startup_trials=30,    
            n_ei_candidates=64,      
            multivariate=True       
            )
    )

    study.optimize(objective, n_trials=120, timeout=20*60*60)

    print("======== Best Trial ========")
    print("params :", study.best_params)
    print("r2_sum :", study.best_value)
    print("r2_V0  :", study.best_trial.user_attrs['r2_V0'])
    print("r2_V1  :", study.best_trial.user_attrs['r2_V1'])
    print("r2_V2  :", study.best_trial.user_attrs['r2_V2'])
    print("r2_std_V0:", study.best_trial.user_attrs['r2_std_V0'])
    print("r2_std_V1:", study.best_trial.user_attrs['r2_std_V1'])
    print("r2_std_V2:", study.best_trial.user_attrs['r2_std_V2'])

    with open(os.path.join(OUT_DIR, "best_params.json"), "w") as f:
        json.dump({"best_trial": study.best_trial.number,
                   "best_epoch":  study.best_trial.user_attrs['best_epoch'],
                   "params": study.best_params,
                   "r2_sum": study.best_value,
                   "r2_V0": study.best_trial.user_attrs['r2_V0'],
                   "r2_V1": study.best_trial.user_attrs['r2_V1'],
                   "r2_V2": study.best_trial.user_attrs['r2_V2'],
                   "r2_std_V0": study.best_trial.user_attrs['r2_std_V0'],
                   "r2_std_V1": study.best_trial.user_attrs['r2_std_V1'],
                   "r2_std_V2": study.best_trial.user_attrs['r2_std_V2']}, f, indent=2)
        

    df = study.trials_dataframe(attrs=("number","value","params","user_attrs"))
    df.to_csv(os.path.join(OUT_DIR, "all_trials.csv"), index=False)
