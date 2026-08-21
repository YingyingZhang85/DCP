import torch
import torch.nn.functional as F
import torch.nn as nn

class DiseaseProgressionLoss(nn.Module):
    def __init__(self, margin=0.2, rank_weight=0.5, mono_weight=0.5, recon_weight=0.1, kl_weight=0.01, anchor_spacing=2.0):
        super(DiseaseProgressionLoss, self).__init__()
        self.margin = margin
        self.rank_weight = rank_weight
        self.mono_weight = mono_weight
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight 
        self.anchor_spacing = anchor_spacing
       
        # Optimization 2: Dynamically generate centers to prevent out-of-bounds.
        # Directly leverage the property that DIAGNOSIS=1,2,3, no base_offset needed
        # centers: [0.0, 2.0, 4.0, 6.0, 8.0 ...]
        # Labels 1, 2, 3 will be mapped to 2.0, 4.0, 6.0 respectively
        max_classes = 10 
        centers = torch.arange(max_classes, dtype=torch.float32) * self.anchor_spacing
        self.register_buffer('prior_centers', centers)

    def forward(self, scores, labels, times, subj_ids, mu, logvar, imgs = None, recon_imgs = None):
        """
        Input tensors are 1D tensors of length N (entire batch).
        scores: Predicted scores (N,)
        labels: Ground truth labels (N,)
        times: Absolute timestamps or age (N,)
        subj_ids: Subject identification numbers (N,)
        imgs: Original input images (N, 1, 91, 109, 91), optional
        recon_imgs: Reconstructed images (N, 1, 91, 109, 91), optional
        
        Returns:
            dict: Dictionary containing total loss and individual sub-losses
        """
        # [Core Addition]: Structured Bayesian KL Divergence
        # 1. Group-prior KL for the Score dimension (dim 0)
        mu_score = mu[:, 0]
        logvar_score = logvar[:, 0]
        
        # Assign target Gaussian centers based on the patient's ground-truth label
        prior_mu = self.prior_centers[labels.long()] # shape: (N,)
        # prior_logvar = torch.zeros_like(prior_mu)    # Variance is 1, log(1) = 0
        
        # KL( q(z_0|x) || N(prior_mu, 1) )
        # Simplified formula
        # Since the target variance is 1 (logvar=0), this simplifies to the same form as standard VAE:
        kl_score = -0.5 * torch.sum(1 + logvar_score - (mu_score - prior_mu).pow(2) - logvar_score.exp())
        
        # 2. Standard normal KL prior N(0,1) for remaining feature dimensions (dims 1-255)
        mu_rest = mu[:, 1:]
        logvar_rest = logvar[:, 1:]
        kl_rest = -0.5 * torch.sum(1 + logvar_rest - mu_rest.pow(2) - logvar_rest.exp())
        
        # Compute average losses separately (logged independently for Wandb monitoring)
        # kl_score: summed over N*1 elements, divided by N -> per-sample
        loss_kl_score = kl_score / mu.size(0)
        # kl_rest: summed over N*255 elements, must divide by N*255 -> per-element, otherwise 255x larger than kl_score
        loss_kl_rest = kl_rest / (mu.size(0) * (mu.size(1) - 1))
        loss_kl = loss_kl_score + loss_kl_rest


        # ==========================================
        # --- Loss 2: Ordinal Ranking Loss ---
        # ==========================================
        # Extract V0 (Baseline) indices for each unique subject
        # Since the dataset is pre-sorted by time, the first occurrence 
        # of a subj_id corresponds to its baseline visit (V0).
        v0_indices = []
        for subj in torch.unique(subj_ids):
            idx = (subj_ids == subj).nonzero(as_tuple=True)[0][0]
            v0_indices.append(idx)
        v0_indices = torch.stack(v0_indices)

        # We only compare baselines to establish a clean inter-subject reference.
        v0_scores = scores[v0_indices]
        v0_labels = labels[v0_indices]

        # Construct M x M benchmark matrix (M = number of unique subjects)
        v0_score_diff = v0_scores.unsqueeze(1) - v0_scores.unsqueeze(0)
        v0_label_diff = v0_labels.unsqueeze(1) - v0_labels.unsqueeze(0)
        
        # Only compare subjects with different clinical labels
        v0_cross_mask = (v0_label_diff != 0)
        
        loss_rank = torch.tensor(0.0, device=scores.device)
        if v0_cross_mask.any():
            dynamic_margin = self.margin * torch.abs(v0_label_diff[v0_cross_mask])
            # Penalty if predicted score difference contradicts baseline label difference
            penalty_rank = F.relu(dynamic_margin - v0_score_diff[v0_cross_mask] * torch.sign(v0_label_diff[v0_cross_mask]))
            loss_rank = torch.mean(penalty_rank)

        
        # ==========================================
        # --- Loss 3: Individual Longitudinal Monotonicity Loss ---
        # ==========================================
        # Rule: Same subject, time progression (time_diff > 0)
        # We use all N samples to enforce the full V0 -> V1 -> V2 trajectory
        score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)   # Shape: (N, N)
        time_diff = times.unsqueeze(1) - times.unsqueeze(0)      # Shape: (N, N)
        is_same_subj = subj_ids.unsqueeze(1) == subj_ids.unsqueeze(0) # Shape: (N, N)

        # Rule: Same subject, forward time progression (time_diff > 0)
        long_mask = is_same_subj & (time_diff > 0)

        loss_mono = torch.tensor(0.0, device=scores.device)
        if long_mask.any():
            # If time_diff > 0, dim 1 is a later visit. 
            # Penalty triggers if the score decreases over time (score_diff < 0).
            inversion_penalty = F.relu(-score_diff[long_mask] + 1e-6)
            # Log-smooth time span amplifier: Increases penalty for longer intervals 
            # while preventing gradient explosion from extreme outliers.
            span_weight = torch.log1p(torch.abs(time_diff[long_mask])) + 1.0
            loss_mono = torch.mean(inversion_penalty * span_weight)


        
        # Compute reconstruction loss (if original and reconstructed images are provided)
        loss_recon = torch.tensor(0.0, device=scores.device)
        if imgs is not None and recon_imgs is not None:
            loss_recon = F.mse_loss(recon_imgs, imgs)
        

        # Total weighted loss
        total_loss = (self.rank_weight * loss_rank) + \
                     (self.mono_weight * loss_mono) + \
                     (self.recon_weight * loss_recon) + \
                     (self.kl_weight * loss_kl) 
        
        # Return loss dictionary for monitoring individual sub-losses
        loss_dict = {
            'total_loss': total_loss,
            'loss_rank': loss_rank,
            'loss_mono': loss_mono,
            'loss_recon': loss_recon,
            'loss_kl': loss_kl,
            'loss_kl_score': loss_kl_score, # Added: independently monitor disease score pulling force
            'loss_kl_rest': loss_kl_rest    # Added: independently monitor background anatomical feature pulling force
        }
        
        return loss_dict
