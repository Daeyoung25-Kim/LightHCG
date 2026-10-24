# LightHCG
Source code of LightHCG model

## Original Paper
D, Kim(2025), "LightHCG: a Lightweight yet powerful HSIC Disentanglement based Causal Glaucoma Detection Model framework", arXiv:2512.02437, pp. 1-11. [arXiv:2512.02437](https://arxiv.org/abs/2512.02437)

## What is LightHCG?

LightHCG is a lightweight, yet powerful latent causal representation learning based glaucoma diagnosis/treatment simulation framework introduced in December, 2025. Based on HSIC disentanglement, which disentangles glaucoma related(causal, Z<sub>2</sub>), non-related features(non-causal, Z<sub>1</sub>) within the extracted latent subspace from retinal fundus images, and a feature-wise Graph Autoencoder-based neural causal discovery algorithm for latent causal features(z<sub>2</sub>), the proposed framework enables vision AI to comprehend underlying causal(physical) mechanisms of glaucoma from retinal images, while intensively blocking the possibility of learning spurious correlations within data which can hinder interpretability and practicality in terms of real-world applications.

<p align="center">
  <img width="451" height="355" alt="image" src="https://github.com/user-attachments/assets/739125af-aa43-4ddf-a8dc-76de56afb0f9" />
</p>

## Key Results (LightHCG v1)

Results show that LightHCG succeeds in extracting valid latent causal representations that can correspond with key causal factors of glaucoma, and in finding the corresponding latent causal graph, while also achieving an extremely lightweight structure by using only 4-CNN layers for image encoding procedures. When compared to representative vision models, such as VGG16, InceptionV3, or MobileNetV2 in downstream tasks, the LightHCG encoder based downstream glaucoma classification model was found to exhibit the highest metric values in Accuracy, Precision and F1-scores for test data images with only using approximately 1% of parameter space compared to conventional vision AI models (ACRIMA dataset). Further comparisons with existing literature also show that LightHCG exhibits extremely competitive, even higher ROC-AUC(=97.13%) scores in glaucoma classification downstream tasks, while only using three causal latent representations extracted from the LightHCG encoder. Focusing on latent causality within medical AI images, LightHCG not only enables robust, yet lightweight downstream glaucoma classification, but also enables visual treatment (intervention) simulations to individual causal factors of glaucoma through latent causality + image decoder correspondence. 

<p align="center">
  <img width="445" height="432" alt="image" src="https://github.com/user-attachments/assets/077ffa56-8269-4705-9eeb-a1cbe63ca9fd" />
</p>

Specific results and model architectures are explained in the following in article: <b>D, Kim(2025), "LightHCG: a Lightweight yet powerful HSIC Disentanglement based Causal Glaucoma Detection Model framework", arXiv:2512.02437, pp. 1-11.</b>

For more specific details, please refer to the preprint article [arXiv:2512.02437](https://arxiv.org/abs/2512.02437)

<b>If you find LightHCG helpful, please cite the original paper: D, Kim(2025), "LightHCG: a Lightweight yet powerful HSIC Disentanglement based Causal Glaucoma Detection Model framework", arXiv:2512.02437, pp. 1-11.</b>

Thank you!
