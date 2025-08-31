# Novel View Synthesis using DDIM Inversion
---
# Methodology
---
![Methodology](arch.png)

<!-- **Overview:**   -->
Given a single reference image $\mathbf{x_{\text{ref}}}$, we first apply DDIM inversion up to $t=600$ to obtain the mean latent z_ref,μ^inv .
This, together with camera intrinsics/extrinsics, class embeddings, and ray information, is fed into our translation network **TUNet**. 
TUNet predicts the target-view mean latent ẑ_tar,μ^inv, which we combine with the corresponding noise component via one of our fusion strategies to form the initial DDIM latent $\tilde{\mathbf{z}}_{tar}^{\text{inv}}$. 
Finally, this latent is sampled by a pre-trained diffusion model to synthesize the novel view image.
