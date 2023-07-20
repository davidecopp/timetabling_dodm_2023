# utils made avialable for the DODMproject2023

This folder contains the utils that we have made available for helping you in conducting your DODMproject2023.

PLease, let us know in case you realize that it could be good (helpful/interesting) to have further kinds of tools made available to you.


## Contents of the folder

README.md                          this file that you are reading
instance_consistency_checker.py    this util spends a shallow test on the consistency of the input data files for a given instance
                                   (it looks for inconsistencies in the data contained in the .stu and the .exm redundand input data files) 
feasibility_checker.py             use this util to asses the feasibility of your feasible solution for a given instance,
                                   or be told which constraint is violated
sol_evaluator.py                   use this util to copute the penality of a feasible solution
full_analysis.py                   essentially does all what the above do in one single run
