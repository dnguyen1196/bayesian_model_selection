INSTRUCTIONS:
To run the program and get all output (from command line and in results folder), simply run:
$ python3 main.py




Task 1
- Plot training set MSE and test set MSE (BOTH!) against gamma
- Compare these values with the true MSE of the true functions
- Why cannot use training set results to select gamma
- How gamma affects error on the test set
- How to explain these effects

Task 2
- Pick 3 gamma values: too small, just right, too large
- Find the learning curve on the 1000-100 dataset ONLY!
- Vary the training size between 10 and 800 samples (not the whole training set)
- How does test set MSE depend on both gamma and training size

Task 3
- Implement the iterative method to find alpha and beta
- Apply to all 5 datasets
- Show how this compares to the results from task 1 in terms of both lambda and mse
- How quality depends on number of examples and features? (This is done by looking
at the first 3 datasets)

Task 4
- For d = 1: 10, run model selection to choose alpha, beta and find log evidence
- Calculate MSE on test set using MAP for prediction
- Also run non-regularized on the same data set and calculate MSE (augmented dataset)
- The log evidence is done on the train data set?
- Plot log evidence and 2 MSE values over d
- Should we choose alpha, beta and d based on log evidence?
- 


Task 1 output
optimal lambda:  8
MSE:  4.15967850948
optimal lambda:  22
MSE:  5.07829980059
optimal lambda:  27
MSE:  4.31557063032
optimal lambda:  75
MSE:  0.389023387713
optimal lambda:  2
MSE:  0.625308842305

