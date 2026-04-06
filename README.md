# DatathonProject
In order to start create an environment with the requirements specified in requirements_explore_env.txt

The instructions and the code are in decoding_notebook.ipynb

The functions used in the notebook are in decoding_notebook_utils.py

In the folder decoding_outputs there are 4 files:
- ER_diffRS_z_response_free_mag_None_15.npz which contains the correlations (performance) obtained from the decoding of ER_diffRS (Expected Reward difference between repeat and switch) around the response. These correlations correspond to the decoding for each time point (200 time points, between [-1,1]s around the response) for each of the 3 subjects
- EU_diffRS_z_response_free_mag_None_15.npz same for the decoding of EU_diffRS (Estimated Uncertainty difference between repeat and switch)
- PE_z_feedback_free_mag_None_15.npz same for the decoding of PE (Prediction Error) around the feedback
- null_model_correlation.npz correlations of the null-model obtained permuting 200 times the traiing set of a normal decoder.
