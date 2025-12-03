# Context-aware risk steering to reduce false positives in risk assessment

## Overview
This repository provides the definitions for 'Imaging Only' context-aware risk refinement framework, its training, and evaluation as applied to predicting the risk of prostate cancer.
Context-aware risk assessment framework uses the most recent medical data (in this case most recent prostate MRI) of a patient to assess an initial risk of disease. It then leverages any prior context (prior MRI exams along with the time intervals between current and prior exams) to refine the risk assessment. The figure below shows the imaging only framework that uses current and prior prostate MRI exams for prostate cancer risk assessment. This risk-steering framework consists of a) a **representation learner** (RL) model,  b) a **risk estimation/prediction** (RE) model, c) a **temporal learner** (TL) model, and d) **risk refinement** (RR) model.

![PCa risk refinement](./figures/Figure1.jpg)

### Representation Learner
The **RL** transforms high-dimensional radiologic imaging data into subject-specific low-dimensional latent representations. It consists of two contrast-specific CNN encoders that, independently, 
transform T2-weighted and diffusion-weighted volumes to a latent representation. A transformer encoder combines these contrast-specific latent representations into a subject-specific representation.
The **RL** model is pretrained using PI-RADS guided contrastive learning such that representations from subjects with similar radiologic risk assessments i.e., max PI-RADS scores 
are pulled closer to one another while those from different risk assessments are pushed away. This helps the model learn to separate low-risk and high-risk representations in the learned latent space.

### Risk Prediction
The risk estimation/prediction model **RE**  predicts risk of clinically significant prostate cancer (PCa) based on the most recent representation. It consists of a series of Dense layer that predict current and future risk of PCa. A baseline layer predicts current estimate of risk. Each additional Dense layer (associated with time t after Baseline) predicts the marginal increase in risk at time t compared to the Baseline risk.

### Temporal Learner and Risk Refinement
The **TL** aggregates temporal history (in the form of an arbitrary number of longitudinal representations distilled from previous visits along with corresponding times of those visits with respect to the current visit) into a change signal to steer the initial risk assessment using **RR** model.


## Installation Instructions

1. Clone the repository
2. Create a Virtual Environment/Conda environment (python = 3.10)
3. Activate the environment
4. Navigate to the code directory and install the required packages specified in requirements.txt
   a. pip install -r requirements.txt
   b. Approximate installation time on Windows PC ~ 4 mins
5. Run evaluation_notebook






