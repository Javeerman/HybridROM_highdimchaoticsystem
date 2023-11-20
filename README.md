# Hybrid_HDCS
Msc thesis project of Jochem Veerman on the hybrid knowledge-based/deep learning reduced order modelling of high dimensional chaotic systems as performed at the Aerospace Engineering department of the TU Delft

# Source code
The folders "Kolmogorov flow, Re = 20" and "Kolmogorov flow, Re = 34" provide the source code for the reduced order models for 2D Kolmogorov flow at a Reynolds number of 20 and 34. For each test, the knowledge-based, deep learning and hybrid reduced order models are provided together with the Bayesian optimisation routine to obtain the hyperparameters of the echo state network. Furhermore, in folder "Reference" the codes to generate the reference data are provided using the KolSol library in Python

# References
The knowledge-based model was obtained from the Msc thesis 
M. Lesjak. Prediction of Chaotic Systems with Physics - Enhanced Machine Learning Models (Master thesis). 2020.

The code for the multi-scale autoencoder was obtained from
A. Racca, N. A. K. Doan, and L. Magri. Modelling spatiotemporal turbulent dynamics with the convolutional autoencoder echo state network. arXiv preprint arXiv:2211.11379, 2022.

The deep learning model was constructed through the codes of 
M. Lesjak. Prediction of Chaotic Systems with Physics - Enhanced Machine Learning Models (Master thesis). 2020.
and 
A. Racca, N. A. K. Doan, and L. Magri. Modelling spatiotemporal turbulent dynamics with the convolutional autoencoder echo state network. arXiv preprint arXiv:2211.11379, 2022.

Finally, the hybrid model code of M. Lesjak. Prediction of Chaotic Systems with Physics - Enhanced Machine Learning Models (Master thesis). 2020. provided the basis for the proposed hybrid model.
