/**
  * Created by xxx on 3/29/2017.
  */
import java.io.{BufferedWriter, FileWriter, File, StringWriter}
import java.util.Random
import au.com.bytecode.opencsv.CSVWriter
import org.apache.commons.math3.distribution.NormalDistribution
import scala.collection.JavaConversions._
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, DenseMatrix => BDM, CSCMatrix => BSM, norm => BNorm, _}
import breeze.numerics._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import scala.util.control.Breaks


object DistributedOL2 {
  /****
    *
    * @param args
    */

  def main(args: Array[String]) {

    for(j<-0 until 1) {

      /** disable LogInfo */
      Logger.getLogger("org").setLevel(Level.ERROR)
      Logger.getLogger("aka").setLevel(Level.ERROR)
      Logger.getLogger("Remoting").setLevel(Level.ERROR)
      Logger.getLogger("spark").setLevel(Level.ERROR)
      Logger.getLogger("spark").setLevel(Level.WARN)
      Logger.getLogger("LAPACK").setLevel(Level.WARN)
      Logger.getLogger("BLAS").setLevel(Level.WARN)

      val conf = new SparkConf().setMaster("local[*]").setAppName("The Ordered Ridge Regression")
      //val conf = new SparkConf().setMaster("spark://spark1:7077").setAppName("The Ordered Ridge Regression") // Used for Cloud
      //val conf = new SparkConf().setAppName("The Ordered Ridge Regression")
      val sc = new SparkContext(conf)

      /** disable LogInfo */
      sc.setLogLevel("ERROR")
      sc.setLogLevel("WARN")
      /****** Load colon data*****/
      val mat = csvread(file=new File("E:/csv/data/A.csv"),separator=',', skipLines=0)
      val lab = csvread(file=new File("E:/csv/data/b.csv"),separator=',', skipLines=0)

      /******** For Cluster ****************/
      //val mat = csvread(file=new File("/home/humayoo/data/A.csv"),separator=',', skipLines=0)
      //val lab = csvread(file=new File("/home/humayoo/data/b.csv"),separator=',', skipLines=0)

      /** **** Generate problem data */
      val m = mat.rows; // number of examples (rows)
      val n = mat.cols; // number of features (Column)
      val q = 0.4;            // False Discovery Rate Parameter
      val spDensdity = breeze.stats.distributions.Rand.gaussian(0,100.0/n.toDouble) // sparsity density
      val A: RDD[BDM[Double]] = sc.parallelize(Seq(mat));
      //A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n); // normalize columns
      val sumofSqrt = A.map { case (a: BDM[Double]) =>
        val r = a :^ 2.0;
        val s = sum(r.t, Axis._1);
        val sr = s :^ 0.5
        val st = sr
        1.0 :/ st;
      }
      val d = sumofSqrt.map { case (x: BDV[Double]) => diag(x) }
      val aColsBrows = A.zip(d);
      val Ardd: RDD[BDM[Double]] = aColsBrows.map { case (colA: BDM[Double], rowB: BDM[Double]) =>
        (colA * rowB)
      }
      val b = BDV(lab.toArray);

      /****************** FDR Formula ************************************************/
      val i = BDV.rangeD(1.0, n+1, 1.0);
      val qi:BDV[Double] = 1.0 - q*i/n.toDouble/2.0
      val standardNormal = new NormalDistribution(0, 1);
      val infNorm = qi.map(x=> standardNormal.inverseCumulativeProbability(x))
      val  lambda:BDV[Double] = infNorm;

      /** ***** Solve problem  ********/
      val result = Ridge(sc, Ardd, b, lambda, 1.0, 1.0);
      /** *
        * length(history.objval)
        */
      val arrayLength = result("objective").filter(x => !x.isNaN).length

      /**
        * Writing Result to Excel File
        */
      val csvFile = new File("E:/csv/" + j + ".csv")
     //val csvFile = new File("/home/humayoo/data/" + j + ".csv") // For Cluster
      val out = new BufferedWriter(new FileWriter(csvFile));
      val writer = new CSVWriter(out);

      /** *
        * Printing Result
        */
      var padding = "%1$3s\t%2$10s\t%3$10s\t%4$10s\t%5$10s\t%6$10s\n"
      var header = padding.format("iter", "r norm", "eps pri", "s norm", "eps dual", "objective")
      println(header)
      writer.writeAll(List(Array("iter", "r norm", "eps pri", "s norm", "eps dual", "objective")))
      for (k <- 0 until arrayLength) {
        padding = "%1$3d\t%2$15.4f\t%3$10.4f\t%4$10.4f\t%5$10.4f\t%6$10.2f\n"
        header = padding.format(k, result("rnorm")(k), result("epspri")(k), result("snorm")(k), result("epsdual")(k), result("objective")(k))
        println(header)
        writer.writeAll(List(Array(k.toString, (result("rnorm")(k)).toString, (result("epspri")(k)).toString, (result("snorm")(k)).toString, (result("epsdual")(k)).toString, (result("objective")(k)).toString)))
      }

      writer.close()
      out.close()
      sc.stop()
    }

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
    */

  def Ridge (sc:SparkContext,A:RDD[BDM[Double]],b:BDV[Double],lambda:BDV[Double],rho:Double,alpha:Double): Map[String, Array[Double]] =
  {
    /*********  Global constants and defaults */

    val QUIET:Int    = 0;
    val MAX_ITER:Int = 1000;
    val ABSTOL:Double   = 1e-4;
    val RELTOL:Double   = 1e-2;

    /*********  Data preprocessing **/
    val Anew:BDM[Double] =   A.reduce(_+_);
    val m:Int = Anew.rows;
    val n:Int = Anew.cols;

    /*********  save a matrix-vector multiply */
    /*
  val Atb:BDV[Double] = A.map{case(a:BDM[Double])=> a.t*b;}.reduce(_+_);
  val AtA:BDM[Double] = A.map{case(a:BDM[Double])=> a.t*a;}.reduce(_+_);
  val AAt:BDM[Double] = A.map{case(a:BDM[Double])=> a*a.t;}.reduce(_+_);
    */
    val Atb:BDV[Double] = Anew.t*b;
    val AtA:BDM[Double] = Anew.t*Anew;
    val AAt:BDM[Double] = Anew*Anew.t;

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
          x = q/rho - (Anew.t*(U \ ( L \ (Anew*q) )))/math.pow(rho,2.0);


        /****** z-update with relaxation ****/
        val zold:BDV[Double] = z;
        val x_hat:BDV[Double] = alpha*x + (1 - alpha)*zold;
        z = Shrinkage(x_hat + u, lambda,rho,n);

        /****  u-update *******/
        u = u + (x_hat - z);
        /**** diagnostics, reporting, termination checks ****/
        objval(k)  = objective(Anew, b, lambda, x, z);

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
