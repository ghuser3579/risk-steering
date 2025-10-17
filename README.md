# Context aware risk steering
This repository provides the basic code for 'Imaging Only' risk refinement model that uses prior imaging context  
This risk refinement framework consists of a) a **representation learner** (RL) model,  b) a **risk estimation/prediction** (RP) model, c) a **temporal learner** (TL) model, and d) **risk refinement** (RR) model.

## Representation Learner
The **RL** transforms clinical data or high-dimensional radiologic imaging data into subject-specific low-dimensional latent representations. It consists of two contrast-specific CNN encoders that, independently, 
transform T2-weighted and diffusion-weighted volumes to a latent representation. A transformer encoder combines the contrast-specific latent representations
into a subject-specific representation.
The **RL** model is trained using PI-RADS guided contrastive learning such that representations from subjects with similar risk assessments i.e., max PI-RADS scores 
are pulled closer to one another while those from different risk assessments are pushed away.

## Risk Prediction
The risk estimation/prediction model predicts risk of clinically significant prostate cancer (PCa) based on the most recent representation. It consists of a series of Dense layer that predict current and future risk of PCa. The baseline layer predicts current estimate of risk. 
Each additional Dense layer (associated with time t after Baseline) predicts the marginal increase in risk at time t compared to the Baseline risk.

## Temporal Learner and Risk Refinement
The **TL** aggregates temporal history (in the form of an arbitrary number of longitudinal representations distilled from previous visits along with corresponding times of those visits with respect to the current visit) into a change signal to steer the initial risk assessment using **RR** model.

![PCa risk refinement](./Figures/Figure1.jpg)


