This repo contains the <ins>appendix</ins> and the <ins>code</ins> for our paper "Disentangled Hyperspherical Clustering for Sepsis Phenotyping".

# Appendix

## Data 
The data was collected under the National Institutes of Health-funded Sepsis Endotyping in Emergency Care (SENECA) project across 12 different UPMC Hospital Sites across South Western Pennsylvania.
The SENECA data contains 43,086 patients in critical care meeting Sepsis-3 criteria. All variables represent max or min value measured during 
the first 6 hours after encounter start time.

**Preprocessing.** Variables with higher than 90\% missingness as well as pure string variables and dates are excluded.  Categorical variables are one-hot encoded and numerical values are normalized. Missing data are inferred by predictive mean matching, a classic algorithm for imputation. Table 1 contains the full features provided to our model and the outcome (mortality during encounter) for evaluation after training.


_Table 1. Summary characteristics of SENECA sepsis data. Features omitted from the
table include the one-hot encodings of the admission hospitals._
![Alt text](figs/demographics.png?raw=true "demographics")


## Training details
**Our method**.  Our model is developed using Pytorch. The experiments are
performed on a machine with a Intel Core 3.30GHzx4 CPU and a GeForce GTX 2080 Ti GPU. The optimizer we selected is Adam. We fix the mini-batch size=<!-- -->128, hidden layer size=<!-- -->128,
learning rate=<!-- -->1e-4, and training epochs=<!-- -->200. Our
hyperparameters *λ*s are selected using grid search from
{1*e*<sup> − 4</sup>, 1*e*<sup> − 3</sup>, 1*e*<sup> − 2</sup>,
1*e*<sup> − 1</sup>, 1*e*<sup>0</sup>, 1*e*<sup>1</sup>}. The optimal
setting has *λ*<sub>*x*, *z*</sub> = 1*e*<sup>1*e* − 1</sup>,
*λ*<sub>∥.∥</sub> = 1*e*<sup>0</sup>,
*λ*<sub>*z*, *c*</sub> = 1*e*<sup>0</sup>,
*λ*<sub>*F*</sub> = 1*e*<sup>1</sup>, *λ*<sub>*x*, *x̂*</sub> = 1*e*0,
$\\Ls\_{entropy}=1e^{-1}$.

**Baselines.** For the autoencoder based methods including DEC, IDEC, DCN and HC, we utilize the same autoencoder backbone, mini-batch, learning rate, and training epoch as those of our method for fair comparison. Their hyerparamters for loss objectives other than the autoencoder based losses are selected from same grid. For AG, GMM, KM++ and CKM, we employ their default hyperparameter selection process built into the algorithms.

## Comparing with the CKM phenotypes
_Figure 1. Boxplots for the limited features from our method in our clusters versus in clusters from CKM. Our method has white background while CKM has yellow background._
![Alt text](figs/clusters.png?raw=true "demographics")