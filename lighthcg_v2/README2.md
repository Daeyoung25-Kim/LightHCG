## LightHCG v2

-Version 1: 2025/12/02

-Version 2: 2026/10/24


## Original Article 

Kim, D. (2025). LightHCG: a Lightweight yet powerful HSIC Disentanglement based Causal Glaucoma Detection Model framework. arXiv preprint arXiv:2512.02437. https://arxiv.org/abs/2512.02437


## Summary

This is the source code of LightHCG_v2, which incorporates a different downstream model definition approach with updated tensorflow packages. (Few changes also applied in hyperparameter settings within the original LightHCG training framework considering environmental changes in tensorflow package). While the first approach (LightHCG_v1) used latent causal representations: Z2 itself for downstream classification, the new version considers uncertainty of latent causal factors by using both $\mu$ and $\sigma$ vectors linked to Z2. Using 6 input features, the new downstream model (LightHCG encoder + DNN) achieved even higher performance in glaucoma detection (AUC score: 98.57%) with extremely less parameter usage (ACRIMA dataset). Below Figures and Tables provide a visual summary of results from LightHCG_v2 ((I) quantitative, qualitative evaluations of latent causal representation learning results, (II) downstream model performance in glaucoma detection).

<p align="center">
  <img width="467" height="382" alt="image" src="https://github.com/user-attachments/assets/91916825-b006-4f2e-9da0-5e1d417ea62c" />
</p>



*Binarized causal adjacency matrix from the extracted latent causal graph exhibits only SHD=1.0 when compared to the ground truth glaucoma causal DAG (top 20% absolute weights converted to 1, otherwise 0)

<p align="center">
  <img width="701" height="727" alt="image" src="https://github.com/user-attachments/assets/06f2720e-c3c1-4e26-9c2d-c3e8d6f45f28" />
</p>


*When varying individual latent causal features in Z2 latent space, one-to-one correspondence to neuroretinal rim (Z2,0) and optic cup (Z2,2) was clearly visible, whereas abnormal vessel correspondence was relatively hard to identify in Z2,1. Considering both SHD metrics and visual disentanglement check results, LightHCG_v2 was found to extract and disentangle valid causal representations of the underlying causal (physical) mechanisms of glaucoma progression.

<p align="center">
  <img width="1170" height="612" alt="image" src="https://github.com/user-attachments/assets/e853f765-cf9c-4fd1-885e-790884dc7962" />
</p>

*Downstream Model exhibits 98.57% AUC score with 95.70% F1-score. Considering use of only 167,431(=163,278+4153) parameters (approximately 1% of representative vision AI models), LightHCG_v2 exhibits extreme performance in both glaucoma detection reliability and efficiency. 

<p align="center">
  <img width="540" height="452" alt="image" src="https://github.com/user-attachments/assets/a73f1fd6-4bb8-4a8c-a864-e0b0bc6b96d2" />
</p>

*Downstream Model Structure: LightHCG encoder + 3-layer DNN (64,32,32 nodes with batch normalization and Elu activation functions, DropOut(0.05) defined between first and second hidden layer)

<p align="center">
  <img width="930" height="355" alt="image" src="https://github.com/user-attachments/assets/c5e11571-a16b-4518-9434-1fab6550c352" />
</p>

For specific information regarding model structures or process of latent causal representation learning for glaucoma progression, please check the original LightHCG_v1 article.

Thank you!
