# Hybrid_HDCS
Msc thesis project of Jochem Veerman on the hybrid knowledge-based/deep learning reduced order modelling of high dimensional chaotic systems as performed at the Aerospace Engineering department of the TU Delft

# Source code
The folders "Kolmogorov flow, Re = 20" and "Kolmogorov flow, Re = 34" provide the source code for the reduced order models for 2D Kolmogorov flow at a Reynolds number of 20 and 34. For each test, the knowledge-based, deep learning and hybrid reduced order models are provided together with the Bayesian optimisation routine to obtain the hyperparameters of the echo state network. Furhermore, in folder "Reference" the codes to generate the reference data are provided using the KolSol library in Python

# References
The knowledge-based model was obtained from 
M. Lesjak and N. A. K. Doan. Chaotic systems learning with hybrid echo state network/ proper orthogonal decomposition based model. Data-Centric Engineering, 2:e16, 2021.

The codes for the multi-scale autoencoder and the Bayesian optimisation was obtained from
A. Racca, N. A. K. Doan, and L. Magri. Modelling spatiotemporal turbulent dynamics with the convolutional autoencoder echo state network. arXiv preprint arXiv:2211.11379, 2022.

The deep learning model was constructed through the codes of 
M. Lesjak. Prediction of Chaotic Systems with Physics - Enhanced Machine Learning Models (Master thesis). 2020.
and 
A. Racca, N. A. K. Doan, and L. Magri. Modelling spatiotemporal turbulent dynamics with the convolutional autoencoder echo state network. arXiv preprint arXiv:2211.11379, 2022.

The hybrid model code of M. Lesjak. Prediction of Chaotic Systems with Physics - Enhanced Machine Learning Models (Master thesis). 2020. provided the basis for the proposed hybrid model.

Finally, the recycle validation method was based on A. Racca and L. Magri. Robust optimization and validation of echo state networks for learning chaotic dynamics. Neural Networks, 142:252–268, 2021
