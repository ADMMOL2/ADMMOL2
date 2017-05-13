import java.io.File
import java.util.Random
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, DenseMatrix => BDM, CSCMatrix => BSM, norm => BNorm, _}
import breeze.numerics.{abs, sqrt}
import breeze.plot._
import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.spark.mllib.linalg.{DenseVector, Vectors, SparseMatrix,Vector}
import org.jfree.chart.axis.LogarithmicAxis
import scala.util.control.Breaks

/**
  * Created by xxx on 3/27/2017.
  */
object OL2 {

  /****
    *
    * @param args
    */

  def main(args: Array[String]) {

    /****** Generate problem data */

    /****** Load colon data*****/
    val mat = csvread(file=new File("data/A.csv"),separator=',', skipLines=0)
    val lab = csvread(file=new File("data/b.csv"),separator=',', skipLines=0)

    val m = mat.rows;       // number of examples (rows)
    val n = mat.cols;       // number of features (Column)
    val q = 0.4;            // False Discovery Rate Parameter
    val spDensdity = breeze.stats.distributions.Rand.gaussian(0,100.0/n.toDouble)  // sparsity density

    var A:BDM[Double] =  mat;
    val i = BDV.rangeD(1.0, n+1, 1.0);
    val sumofSqrt =  diag(1.0:/(sum(A:^2.0,Axis._0):^0.5).t);
    //A = A*sumofSqrt; // this is for normalization of column. but here we do not need in real example.
                      // it takes more itereation step for real. but we use it in numerical
    val b:BDV[Double] = BDV(lab.toArray);
    /****************** FDR Formula ************************************************/
    val qi:BDV[Double] = 1.0 - q*i/n.toDouble/2.0
    //val infNorm = qi.map(x=> inverseCDF(cdf(x)))
    val standardNormal = new NormalDistribution(0, 1);
    val infNorm = qi.map(x=> standardNormal.inverseCumulativeProbability(x))
    val  lambda:BDV[Double] = infNorm;
    /******* Solve problem  ********/
    val result = Ridge(A,b,lambda,1.0,1.0);
    /***
      *length(history.objval)
      */
    val arrayLength = result("objective").filter(x => !x.isNaN).length

    /***
      * Printing Result
      */
    var padding = "%1$3s\t%2$10s\t%3$10s\t%4$10s\t%5$10s\t%6$10s\n"
    var header = padding.format("iter", "r norm", "eps pri", "s norm", "eps dual", "objective")
    println(header)
    for(k <-0 until arrayLength) {
      padding = "%1$3d\t%2$15.4f\t%3$10.4f\t%4$10.4f\t%5$10.4f\t%6$10.2f\n"
      header = padding.format(k, result("rnorm")(k), result("epspri")(k), result("snorm")(k), result("epsdual")(k), result("objective")(k))
      println(header)
    }

    /** Reporting Graph */

    /** Graph is not supporting in BDA.So i comment it */

    val k = arrayLength /** length(history.objval) */
    val h = Figure("Objective Function")
    val p = h.subplot(0)

    val xval = new BDV(BDV.range(0, k).data.map(_.toDouble))

    p += plot(xval, result("objective").slice(0, k), '-', colorcode = "black")
    p.xlabel = "iter (k)"
    p.ylabel = "'f(x^k) + g(z^k)'"
    //h.saveas("~/Applications/IdeaProjects/ADMM/Figure1.png") // save current figure as a .png, eps and pdf also supported
    h.refresh()

    val g = Figure("Primal & Dual Function")
    var pp = g.subplot(2, 1, 0)
    pp += plot(xval, result("rnorm").slice(0, k).map(x => (math.max(1e-8, x))), '-', colorcode = "black")
    pp.logScaleY = true;
    pp += plot(xval, result("epspri").slice(0, k).map(x => (x)), '-', colorcode = "blue")
    pp.logScaleY = true;
    pp.ylabel = "||r||_2"

    pp = g.subplot(2, 1, 1)

    pp += plot(xval, result("snorm").slice(0, k).map(x => (math.max(1e-8, x))), '-', colorcode = "black")
    pp.logScaleY = true;
    pp += plot(xval, result("epsdual").slice(0, k).map(x => (x)), '-', colorcode = "red")
    pp.logScaleY = true;
    pp.ylabel = "||s||_2"
    pp.xlabel = "iter (k)"
    g.refresh()

  }

  /***********
    * Ridge  Solve lasso problem via ADMM

    * [z, history] = lasso(A, b, lambda, rho, alpha);

    * Solves the following problem via ADMM:

    * minimize 1/2*|| Ax - b ||_2^2 + 1/2* \lambda || x ||_2^2

    * The solution is returned in the vector x.

    * history is a structure that contains the objective value, the primal and
    * dual residual norms, and the tolerances for the primal and dual residual
    * norms at each iteration.

    * rho is the augmented Lagrangian parameter.

    * alpha is the over-relaxation parameter (typical values for alpha are
    * between 1.0 and 1.8). 
    * code is inspired by boyd's lasso example at:
    * http://www.stanford.edu/~boyd/papers/admm/lasso/lasso_example.html
    */
  def Ridge (A:BDM[Double],b:BDV[Double],lambda:BDV[Double],rho:Double,alpha:Double): Map[String, Array[Double]] =
  {
    /*********  Global constants and defaults */

    val QUIET:Int    = 0;
    val MAX_ITER:Int = 1000;
    val ABSTOL:Double   = 1e-4;
    val RELTOL:Double   = 1e-2;

    /*********  Data preprocessing **/
    val m:Int = A.rows;
    val n:Int = A.cols;

    /*********  save a matrix-vector multiply */
    val Atb:BDV[Double] = A.t*b;
    val AtA:BDM[Double] = A.t*A;
    val AAt:BDM[Double] = A*A.t;

    /*********  ADMM solver **/
    var x:BDV[Double] = BDV.zeros[Double](n);
    var z:BDV[Double] = BDV.zeros[Double](n);
    var u:BDV[Double] = BDV.zeros[Double](n);
    // cache the factorization
    val LU = Factor(AtA,AAt,rho,m,n); // LU Factorization using conjugate gradient method
  val L:BDM[Double] = LU._1;
    val U:BDM[Double] = LU._2;

    /****************** Result Container Declare ****************/
    val objval = Array.fill[Double](MAX_ITER)(Double.NaN)//new Array[Double](MAX_ITER)
  val r_norm = Array.fill[Double](MAX_ITER)(Double.NaN)//new Array[Double](MAX_ITER)
  val s_norm = Array.fill[Double](MAX_ITER)(Double.NaN)//new Array[Double](MAX_ITER)
  val eps_pri = Array.fill[Double](MAX_ITER)(Double.NaN)//new Array[Double](MAX_ITER)
  val eps_dual = Array.fill[Double](MAX_ITER)(Double.NaN)//new Array[Double](MAX_ITER)
  var resultArray:Map[String, Array[Double]] = Map()//new Array[Double](MAX_ITER)
    /*********** Iteration Start ***********************/

    val loop = new Breaks
    loop.breakable {
      for (k<-0 until MAX_ITER )
      {
        /**** x-update ***/
        val q = Atb + rho*(z - u);    /*** temporary value ***/
        if( m >= n )    /*** if skinny **/
          x = U \ (L \ q);
        else            /*** if fat ***/
          x = q/rho - (A.t*(U \ ( L \ (A*q) )))/math.pow(rho,2.0);


        /****** z-update with relaxation ****/
        val zold:BDV[Double] = z;
        val x_hat:BDV[Double] = alpha*x + (1 - alpha)*zold;
        z = Shrinkage(x_hat + u, lambda,rho,n);

        /****  u-update *******/
        u = u + (x_hat - z);
        /**** diagnostics, reporting, termination checks ****/
        objval(k)  = objective(A, b, lambda, x, z);

        /****
          * calculating Primal residual
          * our case primal_residuall = r^K+1 = x^k+1 - z^k+1
          */
        r_norm(k)  =  BNorm(x - z);

        /****
          * calculating dual residual
          * our case resid_dual = s^K+1 = -p*(z^k+1 - z^k)
          */
        s_norm(k)  = BNorm(-rho*(z - zold));

        /****
          * calculating eps_primal ( tolerance for primal residual)
          * our case eps_primal = sqrt(n) * eps_abs + eps_rel * max(||x||, ||z||)
          */
        eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(BNorm(x), BNorm(-z));

        /****
          * calculating eps_dual
          * eps_dual = sqrt(n) * eps_abs + eps_rel * ||A'y||
          */
        eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*BNorm(rho*u);

        /***
          * Convergence Codintion
          * When convergence condition satisfy, terminate the loop
          */
        if (r_norm(k) < eps_pri(k) && s_norm(k) < eps_dual(k))
          loop.break();

      }

    }
    resultArray += ("objective" -> objval)
    resultArray += ("rnorm" -> r_norm)
    resultArray += ("snorm" -> s_norm)
    resultArray += ("epspri" -> eps_pri)
    resultArray += ("epsdual" -> eps_dual)
    resultArray
  }

  def Shrinkage(x:BDV[Double], lambda:BDV[Double],p:Double,n:Int): BDV[Double] =
  {
    val z:BDV[Double] = (p*x)./(lambda +p);
    z
  }

  /***
    * This is the objective function
    * minimize 1/2*|| Ax - b ||_2^2 + 1/2* \lambda || x ||_2^2
    *
    * @param A
    * @param b
    * @param lambda
    * @param x
    * @param z
    * @return
    */
  def objective(A:BDM[Double], b:BDV[Double], lambda:BDV[Double], x:BDV[Double], z:BDV[Double]):Double =
  {

    //val p:Double = 1/2*sum((A*x - b):^2) + 1/2*(sum(lambda.t*z:^2)); /****  without sorting l2 norm ***/
    val p:Double = (1.0/2.0)*sum((A*x - b):^2.0) + (1.0/2.0)*sum(lambda:*(BDV(z.toArray.sortBy(x=>x).reverse)):^2.0); /*** sorting L2 Norm **/
    //val p:Double = 1.0/2.0*sum((A*x - b):^2.0) + (1.0/2.0)*(sum(lambda :* (BDV(z.toArray.map(x=>abs(x)).sortBy(x=>x).reverse)):^2.0)); /*** sorting L2 Norm with absolute value **/
    p
  }

  /*****
    *
    * **** this function is cholesky factorization of (A^t b+ P*I)
    * @param AtA
    * @param AAt
    * @param rho
    * @param m
    * @param n
    * @return
    */
  def Factor(AtA:BDM[Double],AAt:BDM[Double],rho:Double,m:Int,n:Int):(BDM[Double],BDM[Double])=
  {


    if(m>=n)
    {
      val L:BDM[Double] = BDM.zeros[Double](n,n);
      val cholMat = AtA + rho* BDM.eye[Double](n);
      for (i <- 0 until n) {
        for (j <- 0 to i) {
          var sum: Double = 0.0;
          for (k <- 0 until j) {
            sum += L(i, k) * L(j, k);
          }
          if (i == j)
            L(i, i) = Math.sqrt(cholMat(i, i) - sum);
          else
            L(i, j) = 1.0 / L(j, j) * (cholMat(i, j) - sum);
        }
        if (L(i,i) <= 0) {
          throw new RuntimeException("Matrix not positive definite");
        }
      }

      (L,L.t)
    }
    else
    {
      val L:BDM[Double] = BDM.zeros[Double](m,m);
      val cholMat = BDM.eye[Double](m) + (1.0/rho)*AAt;
      for (i <- 0 until m) {
        for (j <- 0 to i) {
          var sum: Double = 0.0;
          for (k <- 0 until j) {
            sum += L(i, k) * L(j, k);
          }
          if (i == j)
            L(i, i) = Math.sqrt(cholMat(i, i) - sum);
          else
            L(i, j) = 1.0 / L(j, j) * (cholMat(i, j) - sum);
        }
        if (L(i,i) <= 0) {
          throw new RuntimeException("Matrix not positive definite");
        }
      }

      (L,L.t)
    }

  }




}
