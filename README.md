# ML_HMM_re-estimation

Given HMM probability vectors represented in formats of Initial Probability, Transition Probability, and Emission Probabilities, algorithms tabulated in this project takes the ideas behind forward, backward, and forward-backward probability algorithms to maximize the probability of an text observation sequence. Results are returned in the form of tuples of natural log probabilities, and inputs are presented as txt corpus files.

Within relevant methods, relevant integer floating point preservation techniques are implemented, such as log addition and the log-sum-exp technique.

The terminologies of the 3 probabilistic computations are as follows:
1. Forward probability: probability that the partial sequence up to time t has been seen and the current state is i.
2. Backward probability: probability that the partial sequence from t + 1 to T − 1 will be seen given that the current state is i.
3. Forward-Backward Probability: Re-estimated parameters of the native HMM in the form of re-estimated initial, transition, and emission probabilities.

t ranges from 0 to T - 1, where T is the total number of observations or number of text phrases.

* Supplemental methods, such as update(), produce additional parameter calculations for future uses. Update() specifically repeatedly runs forward-bockward() to iteratively re-estimate the HMM probabilities, stopping when the increase in probability falls below the input cutoff threshold value.
