This repo contains the <ins>appendix</ins> and the <ins>code</ins> for our paper "Disentangled Hyperspherical Clustering for Sepsis Phenotyping".

# Appendix

## Hyperspherical Clustering (HC)
HC method builds upon an AE network. During training, an input data
point *x* ∈ ℛ<sup>*m*</sup> is encoded into embedding
*z* ∈ ℛ<sup>*h*</sup> and then decoded back into
*x̂* ∈ *ℛ*<sup>*m*</sup>. From *z* a subnetwork learns clustering
probability *c* ∈ ℛ<sup>*k*</sup>. HC requires of three loss terms. The
first loss ensures that the pairwise feature distances are preserved in
the learned embeddings. Let *δ* be the pairwise cosine similarity and
*γ* be the pairwise euclidean distance. For a minibatch
*X*<sub>*B*</sub>, we define:

ℒ<sub>*x*, *z*</sub> = *λ*<sub>*x*, *z*, 1</sub>(1/*B*)∥*δ*<sub>*B*</sub>(*Z*<sub>*B*</sub>) − *δ*<sub>*B*</sub>(*X*<sub>*B*</sub>)∥<sub>2</sub><sup>2</sup> + *λ*<sub>*x*, *z*, 2</sub>(1/*B*)∥*γ*<sub>*B*</sub>(*Z*<sub>*B*</sub>) − *γ*<sub>*B*</sub>(*X*<sub>*B*</sub>)∥<sub>2</sub><sup>2</sup>.

A soft norm penalty is required to enhance injectivity as it penalizes
embeddings far from the unit norm hypersphere:

ℒ<sub>∥.∥</sub> = max (∥*Z*<sub>*B*</sub>∥<sub>2</sub>, (1 − log (∥*Z*<sub>*B*</sub>∥<sub>2</sub> + 1*e*<sup> − 10</sup>))).

The learned embeddings preserve the angular representation from input
data, so that the unit vectors representing angles have direct
correspondences to cluster probabilities by element-wise square root.
The loss ℒ<sub>*z*, *c*</sub> that encourages matching angles between
embeddings and clustering memberships is defined as:

ℒ<sub>*z*, *c*</sub> = (1/*B*)∥*δ*<sub>*B*</sub>(*C*<sub>*B*</sub><sup>1/2</sup>) − max (*δ*<sub>*B*</sub>(*Z*<sub>*B*</sub>), 0)∥<sub>2</sub><sup>2</sup>.

We also enforce an entropy loss term ℒ<sub>*e**n**t**r**o**p**y*</sub>
that encourages relatively even spreading, and a reconstruction loss
ℒ<sub>*x*, *x̂*</sub> innate to the autoencoder framework. Combining them
with the previously defined losses, we have the total HC loss as:

ℒ<sub>*HC*</sub> = *λ*<sub>*x*, *z*</sub>ℒ<sub>*x*, *z*</sub> + *λ*<sub>∥.∥</sub>ℒ<sub>∥.∥</sub> + *λ*<sub>*z*, *c*</sub>ℒ<sub>*z*, *c*</sub>.


## Predictive-ability-from-few-features Analysis
For DHC, we train a multilayer perceptron (MLP) to predict cluster membership based on the selected features for each pair of clusters. The MLP has two 100 dimensional hidden layers and is trained by Adam for 300 epochs. The train/test split is 80/20.
We calculate 
the average area under the receiver operating characteristic Curve (AUC) score of the test set across all pairs of clusters. 
For each of the clustering methods under comparison, a same number of input features are randomly selected to predict corresponding clusters for 100 random runs, and the average AUC is calculated. 

## Data 
The data was collected under the National Institutes of Health-funded Sepsis Endotyping in Emergency Care (SENECA) project across 12 different UPMC Hospital Sites across South Western Pennsylvania.
The SENECA data contains 43,086 patients in critical care meeting Sepsis-3 criteria. All variables represent max or min value measured during 
the first 6 hours after encounter start time.

**Preprocessing.** Variables with higher than 90\% missingness as well as pure string variables and dates are excluded.  Categorical variables are one-hot encoded and numerical values are normalized. Missing data are inferred by predictive mean matching, a classic algorithm for imputation. Table 1 contains the full features provided to our model and the outcome (mortality during encounter) for evaluation after training.

Features consist of patient demographics, physical exam findngs, and lab results. 
Table 1 contains input features to our model as well as the outcome (mortality during encounter) for evalutaion after training.

_Table 1. Summary characteristics of SENECA sepsis data. Features omitted from the
table include the one-hot encodings of the admission hospitals._
![Alt text](figs/demographics.png?raw=true "demographics")


## Training details
**Our method**.  Our model is developed using Pytorch. The experiments are
performed on a machine with a Intel Core 3.30GHzx4 CPU and a GeForce GTX 2080 Ti GPU. The optimizer we selected is Adam. We fix the mini-batch size=<!-- -->128, hidden layer size=<!-- -->128,
learning rate=<!-- -->1e-4, and training epochs=<!-- -->200. Our
hyperparameters *λ*s are selected using grid search from
{1*e*<sup> − 4</sup>, 1*e*<sup> − 3</sup>, 1*e*<sup> − 2</sup>,
1*e*<sup> − 1</sup>, 1*e*<sup>0</sup>, 1*e*<sup>1</sup>}. 
The optimal
setting has *λ*<sub>*x*, *z*, *1*</sub> = 1*e*<sup>−1</sup>,
 *λ*<sub>*x*, *z*, *2*</sub> = 1*e*<sup>−5</sup>,
*λ*<sub>∥.∥</sub> = 1*e*<sup>0</sup>,
*λ*<sub>*z*, *c*</sub> = 1*e*<sup>0</sup>,
*λ*<sub>*F*</sub> = 1*e*<sup>1</sup>, *λ*<sub>*x*, *x̂*</sub> = 1*e*<sup>0</sup>,
*λ*<sub>*Entropy*</sub>=1*e*-1.

**Baselines.** For the autoencoder based methods including DEC, IDEC, DCN and HC, we utilize the same autoencoder backbone, mini-batch, learning rate, and training epoch as those of our method for fair comparison. Their hyerparamters for loss objectives other than the autoencoder based losses are selected from same grid. For AG, GMM, KM++ and CKM, we employ their default hyperparameter selection process built into the algorithms.

## Comparing with the CKM phenotypes
_Figure 1. Boxplots for the limited features from our method in our clusters versus in clusters from CKM. Our method has white background while CKM has yellow background._
![Alt text](figs/clusters.png?raw=true "demographics")