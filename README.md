- First run setup.py to download and set up the directories
- After successful completion, a directory named 15CS10050_Assignment2 is created which contains all the required files
- Copy data folder into this directory
- Then you can run any of the python files present in codes directory with flag --train to train the network or flag --test to test the experiments
    - e.g. python3 15CS10050_Assignment2_task_b_expt_2.py --train 
- Before running test for any experiment, check if the corresponding weights are present in the weights directory

- VERY IMPORTANT - RUN TASK (A) BEFORE RUNNING ANY OTHER TASKS

- Plots get saved in the plots directory

- Required packages:
    - numpy
    - mxnet
    - matplotlib
    - zipfile

- Directory structure
    - 15CS10050_Assignment2
        - code
        - weights
        - data
        - plots
        - report.pdf
        - README.txt
        - setup.py


You can run 
python3 15CS10050_Assignment2_task_a.py --test; python3 15CS10050_Assignment2_task_b_expt_1.py --test; python3 15CS10050_Assignment2_task_b_expt_2.py --test; python3 15CS10050_Assignment2_task_b_expt_3.py --test; python3 15CS10050_Assignment2_task_c.py --test 
to test all experiments
