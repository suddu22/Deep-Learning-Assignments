- First run setup.py to download weights
- Link to git repo: https://github.com/suddu22/Deep-Learning-Assignments.git
- After successful completion, weights directory is added which contains all the weight files
- Copy data folder into the root directory
- cd into code directory
- Then you can run any of the python files present in code directory with flag --train to train the network or flag --test to test the experiments
    - e.g. python3 15CS10050_Assignment2_task_b_expt_2.py --train 
- Before running test for any experiment, check if the corresponding weights are present in the weights directory

- VERY IMPORTANT - RUN TASK (A) BEFORE RUNNING ANY OTHER TASKS FOR TRAINING

- Plots get saved in the plots directory

- Required packages:
    - numpy
    - mxnet
    - matplotlib
    - zipfile
    - shutil
    - requests

- Directory structure
    - 15CS10050_Assignment2
        - code
        - weights
        - data
        - plots
        - report.pdf
        - README.txt
        - setup.py


- You can run 
python3 15CS10050_Assignment2_task_a.py --test; python3 15CS10050_Assignment2_task_b_expt_1.py --test; python3 15CS10050_Assignment2_task_b_expt_2.py --test; python3 15CS10050_Assignment2_task_b_expt_3.py --test; python3 15CS10050_Assignment2_task_c.py --test 
in code directory to test all experiments
