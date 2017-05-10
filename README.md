ADMM-OL2: The ordered L2-Norm minimization
------------------------------------------------------
1.0 Introduction
--------------

Thank you for downloading the ADMM-OL2 solver!  ADMM-OL2 is a apache spark solver
using Scala for large-scale problem based on ADMM. ADMM-OL2 is adaptive, computationally tractable
and distributed. It is designed to solve the following three problems: 

1. The Ordered Ridge Regression Problem:
   minimize  ||Ax - b||_2  subject to  J_λ(x)

Where J_λ(x) is the ordered L2-norm, the matrix A and vector b can be defined explicily, x is unknown.



2.0 Figures
--------------

All figures are drawn using matlab. There is a figure folder which contains all matlab file for drawing
figures. Data to draw figures are availabe in data folder which is inside figure folder. Each matlab file
draw one figure when you run it.

3.0 Spark
--------------
ADMM-OL2 is implemented using Scala and spark. Spark folder contains source code of ADMM-OL2 method.
it contains the following three source code file.

3.1  NumericalOL2.scala
-----------------------
It show the convergence of ADMM-OL2 method.

3.2 OL2.scala
--------------
It is standalone version of ADMM-OL2 method.

3.3 DistributedOL2.scala
------------------------
It is distributed version of ADMM-OL2 method.

data folder contains real colon dataset which is used for experiments.
