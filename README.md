# Novel View Synthesis using DDIM Inversion
---
# Methadology
---
![Methodology Architecture](arch.png)

**Overview:**  
Given a single reference image $\mathbf{x_{\text{ref}}}$, we first apply DDIM inversion up to \(t=600\) to obtain the mean latent \(\mathbf{z}_{\text{ref},\mu}^{\text{inv}}\). This, together with camera intrinsics/extrinsics, class embeddings, and ray information, is fed into our translation network **TUNet**. TUNet predicts the target-view mean latent \(\tilde{\mathbf{z}}_{\text{tar},\mu}^{\text{inv}}\), which we combine with the corresponding noise component via one of our fusion strategies to form the initial DDIM latent \(\tilde{\mathbf{z}}_{tar}^{\text{inv}}\). Finally, this latent is sampled by a pre-trained diffusion model to synthesize the novel view image.
