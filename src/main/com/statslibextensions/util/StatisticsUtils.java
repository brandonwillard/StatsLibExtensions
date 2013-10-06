package com.statslibextensions.util;

import gov.sandia.cognition.math.LogMath;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.matrix.mtj.AbstractMTJMatrix;
import gov.sandia.cognition.math.matrix.mtj.DenseMatrixFactoryMTJ;
import gov.sandia.cognition.math.matrix.mtj.DiagonalMatrixFactoryMTJ;
import gov.sandia.cognition.math.matrix.mtj.DiagonalMatrixMTJ;
import gov.sandia.cognition.math.matrix.mtj.decomposition.SingularValueDecompositionMTJ;
import gov.sandia.cognition.statistics.distribution.ChiSquareDistribution;
import gov.sandia.cognition.statistics.distribution.InverseWishartDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.util.Weighted;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;

import no.uib.cipr.matrix.DenseCholesky;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.UpperSPDDenseMatrix;

import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;

public class StatisticsUtils {

  private static final double[] a = {2.2352520354606839287, 161.02823106855587881,
      1067.6894854603709582, 18154.981253343561249, 0.065682337918207449113};

  private static final double[] b = {47.20258190468824187, 976.09855173777669322,
      10260.932208618978205, 45507.789335026729956};

  private static final double[] c = {0.39894151208813466764, 8.8831497943883759412,
      93.506656132177855979, 597.27027639480026226, 2494.5375852903726711, 6848.1904505362823326,
      11602.651437647350124, 9842.7148383839780218, 1.0765576773720192317e-8};

  private static final int CUTOFF = 16; /* Cutoff allowing exact "*" and "/" */

  private static final double[] d = {22.266688044328115691, 235.38790178262499861,
      1519.377599407554805, 6485.558298266760755, 18615.571640885098091, 34900.952721145977266,
      38912.003286093271411, 19685.429676859990727};

  private static final double DBL_EPSILON = 2.2204460492503131e-016;
  private static final double M_1_SQRT_2PI = 0.398942280401432677939946059934;

  private static final double M_SQRT_32 = 5.656854249492380195206754896838; /* The square root of 32 */

  static final public double MACHINE_EPS = StatisticsUtils.determineMachineEpsilon();

  private static final double[] p_ = {0.21589853405795699, 0.1274011611602473639,
      0.022235277870649807, 0.001421619193227893466, 2.9112874951168792e-5, 0.02307344176494017303};

  private static final double[] q = {1.28426009614491121, 0.468238212480865118,
      0.0659881378689285515, 0.00378239633202758244, 7.29751555083966205e-5};

  private static double determineMachineEpsilon() {
    final double d1 = 1.3333333333333333d;
    double d3;
    double d4;

    for (d4 = 0.0d; d4 == 0.0d; d4 = Math.abs(d3 - 1.0d)) {
      final double d2 = d1 - 1.0d;
      d3 = d2 + d2 + d2;
    }

    return d4;
  }

  /**
   * Returns, for A, an R s.t. R^T * R = A
   * 
   * @param matrix
   * @return
   */
  public static Matrix getCholR(Matrix matrix) {
    final DenseCholesky cholesky =
        DenseCholesky.factorize(DenseMatrixFactoryMTJ.INSTANCE.copyMatrix(matrix)
            .getInternalMatrix());

    final Matrix covSqrt =
        DenseMatrixFactoryMTJ.INSTANCE.createWrapper(new no.uib.cipr.matrix.DenseMatrix(cholesky
            .getU()));

    assert covSqrt.transpose().times(covSqrt).equals(matrix, 1e-4);

    return covSqrt;
  }

  public static Matrix getDiagonalInverse(Matrix S, double lowerTolerance) {
    Preconditions.checkArgument(lowerTolerance >= 0d);
    final Matrix result =
        MatrixFactory.getDefault().createMatrix(S.getNumColumns(), S.getNumColumns());
    for (int i = 0; i < Math.min(S.getNumColumns(), S.getNumRows()); i++) {
      final double sVal = S.getElement(i, i);
      if (Math.abs(sVal) > lowerTolerance) result.setElement(i, i, 1d / sVal);
    }
    return result;
  }


  public static Matrix getDiagonalSqrt(Matrix mat, double tolerance) {
    Preconditions.checkArgument(tolerance > 0d);
    final Matrix result = mat.clone();
    for (int i = 0; i < Math.min(result.getNumColumns(), result.getNumRows()); i++) {
      final double sqrt = Math.sqrt(result.getElement(i, i));
      if (sqrt > tolerance) result.setElement(i, i, sqrt);
    }
    return result;
  }

  public static Matrix getDiagonalSquare(Matrix S, double tolerance) {
    Preconditions.checkArgument(tolerance >= 0d);
    final Matrix result =
        MatrixFactory.getDefault().createMatrix(S.getNumColumns(), S.getNumColumns());
    for (int i = 0; i < Math.min(S.getNumColumns(), S.getNumRows()); i++) {
      final double sVal = S.getElement(i, i);
      final double sValSq = sVal * sVal;
      if (Math.abs(sValSq) > tolerance) result.setElement(i, i, sValSq);
    }
    return result;
  }

  public static Matrix getInvWishartVar(InverseWishartDistribution invWishart) {
    final int dim = invWishart.getInputDimensionality();
    final int dof = invWishart.getDegreesOfFreedom();
    Preconditions.checkArgument(dof > dim - 1);
    final Matrix invScale = invWishart.getInverseScale();
    final Matrix cov = MatrixFactory.getDefault().createMatrix(dim, dim);
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        final double numerator =
            (dof - dim + 1d) * invScale.getElement(i, j) + (dof - dim - 1d)
                * invScale.getElement(i, i) * invScale.getElement(j, j);
        final double denominator =
            (dof - dim) * (dof - dim - 1d) * (dof - dim - 1d) * (dof - dim - 3d);
        cov.setElement(i, j, numerator / denominator);
      }
    }
    return cov;
  }

  public static Matrix getPseudoInverseReduced(Matrix matrix) {
    final SingularValueDecompositionMTJ svd = SingularValueDecompositionMTJ.create(matrix);

    final int effRank = svd.effectiveRank(1e-9);

    final DiagonalMatrixMTJ roots =
        DiagonalMatrixFactoryMTJ.INSTANCE.createMatrix(matrix.getNumColumns());
    final Matrix U = svd.getU();
    for (int i = 0; i < effRank; i++) {
      roots.setElement(i, 1d / svd.getS().getElement(i, i));
    }

    final Matrix firstHalf = U.times(roots).getSubMatrix(0, U.getNumRows() - 1, 0, effRank - 1);

    Matrix V = svd.getVtranspose().transpose();
    V = V.getSubMatrix(0, V.getNumRows() - 1, 0, effRank - 1);
    final Matrix result = V.times(firstHalf.transpose());

    return result;
  }

  /**
   * Uses the underlying arrays in these vector objects for a hash code.
   */
  public static int hashCodeVector(Vector vec) {
    if (vec instanceof gov.sandia.cognition.math.matrix.mtj.DenseVector) {
      return Arrays.hashCode(((gov.sandia.cognition.math.matrix.mtj.DenseVector) vec).getArray());
    } else {
      return vec.hashCode();
    }
  }

  public static double logEvaluateNormal(Vector input, Vector mean, Matrix cov) {
    Preconditions.checkArgument(input.getDimensionality() == mean.getDimensionality());
    final int k = mean.getDimensionality();
    final double logLeadingCoefficient =
        (-0.5 * k * MultivariateGaussian.LOG_TWO_PI) + (-0.5 * cov.logDeterminant().getRealPart());

    final Vector delta = input.minus(mean);
    final DenseVector b = new DenseVector(cov.getNumRows());
    final DenseVector d =
        new DenseVector(((gov.sandia.cognition.math.matrix.mtj.DenseVector) delta).getArray(),
            false);
    final UpperSPDDenseMatrix spd =
        new UpperSPDDenseMatrix(((AbstractMTJMatrix) cov).getInternalMatrix(), false);
    spd.transSolve(d, b);
    final double zsquared = b.dot(d);
    final double result = logLeadingCoefficient - 0.5 * zsquared;

    return result;
  }

  /**
   * Low variance sampler. Follows Thrun's example in Probabilistic Robots.
   */
  public static <SupportType extends Weighted> Multiset<SupportType> lowVarianceSampler(Random rng,
      Multiset<SupportType> particles, double M) {
    Preconditions.checkArgument(particles.size() > 0);

    final Multiset<SupportType> resampled = HashMultiset.create((int) M);
    final double r = rng.nextDouble() / M;
    final Iterator<SupportType> pIter = particles.iterator();
    SupportType p = pIter.next();
    double c = p.getWeight() - Math.log(particles.count(p));
    for (int m = 0; m < M; ++m) {
      final double U = Math.log(r + m / M);
      while (U > c && pIter.hasNext()) {
        p = pIter.next();
        c = LogMath.add(p.getWeight() - Math.log(particles.count(p)), c);
      }
      resampled.add(p);
    }

    return resampled;
  }

  /**
   * Taken from NormalDistribution.java Copyright (C) 2002-2006 Alexei Drummond and Andrew Rambaut
   * 
   * A more accurate and faster implementation of the cdf (taken from function pnorm in the R
   * statistical language) This implementation has discrepancies depending on the programming
   * language and system architecture In Java, returned values become zero once z reaches -37.5193
   * exactly on the machine tested In the other implementation, the returned value 0 at about z = -8
   * In C, this 0 value is reached approximately z = -37.51938
   * 
   * Will later need to be optimised for BEAST
   * 
   * @param x argument
   * @param mu mean
   * @param sigma standard deviation
   * @param log_p is p logged
   * @return cdf at x
   */
  public static double normalCdf(double x, double mu, double sigma, boolean log_p) {
    final boolean i_tail = false;
    double p, cp = Double.NaN;

    if (Double.isNaN(x) || Double.isNaN(mu) || Double.isNaN(sigma)) {
      return Double.NaN;
    }
    if (Double.isInfinite(x) && mu == x) { /* x-mu is NaN */
      return Double.NaN;
    }
    if (sigma <= 0) {
      if (sigma < 0) {
        return Double.NaN;
      }
      return (x < mu) ? 0.0 : 1.0;
    }
    p = (x - mu) / sigma;
    if (Double.isInfinite(p)) {
      return (x < mu) ? 0.0 : 1.0;
    }
    x = p;
    if (Double.isNaN(x)) {
      return Double.NaN;
    }

    double xden, xnum, temp, del, eps, xsq, y;
    int i;
    boolean lower, upper;
    eps = StatisticsUtils.DBL_EPSILON * 0.5;
    lower = !i_tail;
    upper = i_tail;

    y = Math.abs(x);
    if (y <= 0.67448975) { /* Normal.quantile(3/4, 1, 0) = 0.67448975 */
      if (y > eps) {
        xsq = x * x;
        xnum = StatisticsUtils.a[4] * xsq;
        xden = xsq;
        for (i = 0; i < 3; i++) {
          xnum = (xnum + StatisticsUtils.a[i]) * xsq;
          xden = (xden + StatisticsUtils.b[i]) * xsq;
        }
      } else {
        xnum = xden = 0.0;
      }
      temp = x * (xnum + StatisticsUtils.a[3]) / (xden + StatisticsUtils.b[3]);
      if (lower) {
        p = 0.5 + temp;
      }
      if (upper) {
        cp = 0.5 - temp;
      }
      if (log_p) {
        if (lower) {
          p = Math.log(p);
        }
        if (upper) {
          cp = Math.log(cp);
        }
      }
    }

    else if (y <= StatisticsUtils.M_SQRT_32) {
      /* Evaluate pnorm for 0.67448975 = Normal.quantile(3/4, 1, 0) < |x| <= sqrt(32) ~= 5.657 */

      xnum = StatisticsUtils.c[8] * y;
      xden = y;
      for (i = 0; i < 7; i++) {
        xnum = (xnum + StatisticsUtils.c[i]) * y;
        xden = (xden + StatisticsUtils.d[i]) * y;
      }
      temp = (xnum + StatisticsUtils.c[7]) / (xden + StatisticsUtils.d[7]);

      xsq = ((int) (y * StatisticsUtils.CUTOFF)) * 1.0 / StatisticsUtils.CUTOFF;
      del = (y - xsq) * (y + xsq);
      if (log_p) {
        p = (-xsq * xsq * 0.5) + (-del * 0.5) + Math.log(temp);
        if ((lower && x > 0.0) || (upper && x <= 0.0)) {
          cp = Math.log(1.0 - Math.exp(-xsq * xsq * 0.5) * Math.exp(-del * 0.5) * temp);
        }
      } else {
        p = Math.exp(-xsq * xsq * 0.5) * Math.exp(-del * 0.5) * temp;
        cp = 1.0 - p;
      }

      if (x > 0.0) {
        temp = p;
        if (lower) {
          p = cp;
        }
        cp = temp;
      }
    }
    /*
     * else |x| > sqrt(32) = 5.657 : the next two case differentiations were really for lower=T,
     * log=F Particularly *not* for log_p ! Cody had (-37.5193 < x && x < 8.2924) ; R originally had
     * y < 50 Note that we do want symmetry(0), lower/upper -> hence use y
     */
    else if (log_p || (lower && -37.5193 < x && x < 8.2924)
        || (upper && -8.2924 < x && x < 37.5193)) {

      /* Evaluate pnorm for x in (-37.5, -5.657) union (5.657, 37.5) */
      xsq = 1.0 / (x * x);
      xnum = StatisticsUtils.p_[5] * xsq;
      xden = xsq;
      for (i = 0; i < 4; i++) {
        xnum = (xnum + StatisticsUtils.p_[i]) * xsq;
        xden = (xden + StatisticsUtils.q[i]) * xsq;
      }
      temp = xsq * (xnum + StatisticsUtils.p_[4]) / (xden + StatisticsUtils.q[4]);
      temp = (StatisticsUtils.M_1_SQRT_2PI - temp) / y;

      // do_del(x);
      xsq = ((int) (x * StatisticsUtils.CUTOFF)) * 1.0 / StatisticsUtils.CUTOFF;
      del = (x - xsq) * (x + xsq);
      if (log_p) {
        p = (-xsq * xsq * 0.5) + (-del * 0.5) + Math.log(temp);
        if ((lower && x > 0.0) || (upper && x <= 0.0)) {
          cp = Math.log(1.0 - Math.exp(-xsq * xsq * 0.5) * Math.exp(-del * 0.5) * temp);
        }
      } else {
        p = Math.exp(-xsq * xsq * 0.5) * Math.exp(-del * 0.5) * temp;
        cp = 1.0 - p;
      }
      // swap_tail;
      if (x > 0.0) {
        temp = p;
        if (lower) {
          p = cp;
        }
        cp = temp;
      }
    } else { /* no log_p , large x such that probs are 0 or 1 */
      if (x > 0) {
        p = 1.0;
        cp = 0.0;
      } else {
        p = 0.0;
        cp = 1.0;
      }
    }
    return p;

  }

  public static Matrix rootOfSemiDefinite(Matrix matrix) {
    return StatisticsUtils.rootOfSemiDefinite(matrix, false, -1);
  }

  public static Matrix rootOfSemiDefinite(Matrix matrix, boolean effRankDimResult, int rank) {
    final SingularValueDecompositionMTJ svd = SingularValueDecompositionMTJ.create(matrix);

    final int effRank = rank > 0 ? rank : svd.effectiveRank(1e-9);

    final DiagonalMatrixMTJ roots =
        DiagonalMatrixFactoryMTJ.INSTANCE.createMatrix(matrix.getNumColumns());
    final Matrix U = svd.getU();
    for (int i = 0; i < effRank; i++) {
      roots.setElement(i, Math.sqrt(svd.getS().getElement(i, i)));
    }

    Matrix result = U.times(roots);
    if (effRankDimResult) {
      result = result.getSubMatrix(0, result.getNumRows() - 1, 0, effRank - 1);
    }

    return result;
  }

  public static Matrix sampleInvWishart(InverseWishartDistribution invWish, Random rng) {
    final int p = invWish.getInverseScale().getNumRows();
    final Vector Zdiag = VectorFactory.getDenseDefault().createVector(p);
    int ii = 0;
    for (int i = invWish.getDegreesOfFreedom(); i >= invWish.getDegreesOfFreedom() - p + 1; i--) {
      final double chiSmpl = Math.sqrt(ChiSquareDistribution.sample(i, rng, 1).get(0));
      Zdiag.setElement(ii, chiSmpl);
      ii++;
    }
    final Matrix Z = MatrixFactory.getDenseDefault().createDiagonal(Zdiag);

    for (int i = 0; i < p; i++) {
      for (int j = i + 1; j < p; j++) {
        final double normSample = rng.nextGaussian();
        Z.setElement(i, j, normSample);
      }
    }

    final Matrix scaleSqrt =
        StatisticsUtils.rootOfSemiDefinite(invWish.getInverseScale().pseudoInverse(1e-9));
    final Matrix k = Z.times(scaleSqrt);
    final Matrix wishSample = k.transpose().times(k);
    final Matrix invWishSample = wishSample.pseudoInverse(1e-9);
    return invWishSample;
  }

  public static int sum(int[] array) {
    int sum = 0;
    for (int i = 0; i < array.length; i++) {
      sum += array[i];
    }
    return sum;
  }

  /**
   * Uses the underlying arrays in these vector objects for comparison.
   */
  public static boolean vectorEquals(Vector vec1, Vector vec2) {
    if ((vec1 instanceof gov.sandia.cognition.math.matrix.mtj.DenseVector)
        && (vec2 instanceof gov.sandia.cognition.math.matrix.mtj.DenseVector)) {
      return Arrays.equals(((gov.sandia.cognition.math.matrix.mtj.DenseVector) vec1).getArray(),
          ((gov.sandia.cognition.math.matrix.mtj.DenseVector) vec2).getArray());
    } else {
      return Objects.equal(vec1, vec2);
    }
  }
}
