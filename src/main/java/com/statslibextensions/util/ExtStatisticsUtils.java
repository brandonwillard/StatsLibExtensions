package com.statslibextensions.util;

import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.matrix.mtj.AbstractMTJMatrix;
import gov.sandia.cognition.math.matrix.mtj.DenseMatrixFactoryMTJ;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.distribution.ChiSquareDistribution;
import gov.sandia.cognition.statistics.distribution.InverseWishartDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;

import java.util.Comparator;
import java.util.Map.Entry;
import java.util.Random;

import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.NotConvergedException;
import no.uib.cipr.matrix.SymmDenseEVD;
import no.uib.cipr.matrix.UpperSPDDenseMatrix;
import no.uib.cipr.matrix.UpperSymmDenseMatrix;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multiset;
import com.google.common.collect.TreeMultiset;
import com.google.common.primitives.Doubles;
import com.statslibextensions.math.MutableDoubleCount;
import com.statslibextensions.math.matrix.SvdMatrix;

public class ExtStatisticsUtils {

  private static class NormalCDFConstants {
    public static final double[] a = {2.2352520354606839287, 161.02823106855587881,
        1067.6894854603709582, 18154.981253343561249, 0.065682337918207449113};
  
    public static final double[] b = {47.20258190468824187, 976.09855173777669322,
        10260.932208618978205, 45507.789335026729956};
  
    public static final double[] c = {0.39894151208813466764, 8.8831497943883759412,
        93.506656132177855979, 597.27027639480026226, 2494.5375852903726711, 6848.1904505362823326,
        11602.651437647350124, 9842.7148383839780218, 1.0765576773720192317e-8};
  
    public static final int CUTOFF = 16; /* Cutoff allowing exact "*" and "/" */
  
    public static final double[] d = {22.266688044328115691, 235.38790178262499861,
        1519.377599407554805, 6485.558298266760755, 18615.571640885098091, 34900.952721145977266,
        38912.003286093271411, 19685.429676859990727};
  
    public static final double DBL_EPSILON = 2.2204460492503131e-016;
    public static final double M_1_SQRT_2PI = 0.398942280401432677939946059934;
  
    public static final double M_SQRT_32 = 5.656854249492380195206754896838; /* The square root of 32 */

    public static final double[] p_ = {0.21589853405795699, 0.1274011611602473639,
        0.022235277870649807, 0.001421619193227893466, 2.9112874951168792e-5, 0.02307344176494017303};
  
    public static final double[] q = {1.28426009614491121, 0.468238212480865118,
        0.0659881378689285515, 0.00378239633202758244, 7.29751555083966205e-5};
  }

  static final public double MACHINE_EPS = ExtStatisticsUtils.determineMachineEpsilon();

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
   * Computes the variance for an inverse wishart distribution.
   * 
   * @param invWishart
   * @return
   */
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

  /**
   * Evaluate a multivariate normal pdf using the MTJ symmetric matrix
   * <code>solve</code> method.
   * @param input
   * @param mean
   * @param cov
   * @return
   */
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
    eps = NormalCDFConstants.DBL_EPSILON * 0.5;
    lower = !i_tail;
    upper = i_tail;

    y = Math.abs(x);
    if (y <= 0.67448975) { /* Normal.quantile(3/4, 1, 0) = 0.67448975 */
      if (y > eps) {
        xsq = x * x;
        xnum = NormalCDFConstants.a[4] * xsq;
        xden = xsq;
        for (i = 0; i < 3; i++) {
          xnum = (xnum + NormalCDFConstants.a[i]) * xsq;
          xden = (xden + NormalCDFConstants.b[i]) * xsq;
        }
      } else {
        xnum = xden = 0.0;
      }
      temp = x * (xnum + NormalCDFConstants.a[3]) / (xden + NormalCDFConstants.b[3]);
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

    else if (y <= NormalCDFConstants.M_SQRT_32) {
      /* Evaluate pnorm for 0.67448975 = Normal.quantile(3/4, 1, 0) < |x| <= sqrt(32) ~= 5.657 */

      xnum = NormalCDFConstants.c[8] * y;
      xden = y;
      for (i = 0; i < 7; i++) {
        xnum = (xnum + NormalCDFConstants.c[i]) * y;
        xden = (xden + NormalCDFConstants.d[i]) * y;
      }
      temp = (xnum + NormalCDFConstants.c[7]) / (xden + NormalCDFConstants.d[7]);

      xsq = ((int) (y * NormalCDFConstants.CUTOFF)) * 1.0 / NormalCDFConstants.CUTOFF;
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
      xnum = NormalCDFConstants.p_[5] * xsq;
      xden = xsq;
      for (i = 0; i < 4; i++) {
        xnum = (xnum + NormalCDFConstants.p_[i]) * xsq;
        xden = (xden + NormalCDFConstants.q[i]) * xsq;
      }
      temp = xsq * (xnum + NormalCDFConstants.p_[4]) / (xden + NormalCDFConstants.q[4]);
      temp = (NormalCDFConstants.M_1_SQRT_2PI - temp) / y;

      // do_del(x);
      xsq = ((int) (x * NormalCDFConstants.CUTOFF)) * 1.0 / NormalCDFConstants.CUTOFF;
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

  /**
   * Sample from an inverse wishart distribution.  Uses chi-square samples.
   * 
   * @param invWish
   * @param rng
   * @return
   */
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
        ExtMatrixUtils.rootOfSemiDefinite(invWish.getInverseScale().pseudoInverse(1e-9));
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
  
  public static <D> String prettyPrintDistribution(DataDistribution<D> dist) {
    Multiset<Entry<D, ? extends Number>> orderedDist = TreeMultiset.create(
        new Comparator<Entry<D, ? extends Number>>() {
          @Override
          public int compare(Entry<D, ? extends Number> o1,
              Entry<D, ? extends Number> o2) {
//            return o1.getValue().doubleValue() < o2.getValue().doubleValue() ? -1 : 1;
            return -Double.compare(o1.getValue().doubleValue(), 
                o2.getValue().doubleValue());
          }
        });
    orderedDist.addAll(dist.asMap().entrySet());
    StringBuffer sb = new StringBuffer();
    for (Entry<D, ? extends Number> obj : orderedDist.elementSet()) {
      final int setCount = orderedDist.count(obj);
      sb.append(dist.getFraction(obj.getKey()) * setCount);
      sb.append(" (");
      sb.append(dist.getLogFraction(obj.getKey()) + Math.log(setCount));
      sb.append(")");
      if (obj.getValue() instanceof MutableDoubleCount) {
        sb.append(" [" + ((MutableDoubleCount) obj.getValue()).count + "]");
      }
      sb.append(" =\n\t" + obj.getKey().toString());
      if (setCount > 1)
        sb.append("\n...(only displaying first result of " + setCount + ")");
      sb.append("\n");
    }
    return sb.toString();
  }
  
  /**
   * Returns a ~99% confidence interval/credibility region by using the largest
   * eigen value for a normal covariance.
   * 
   * @param covar
   * @return
   */
  public static double getLargeNormalCovRadius(Matrix covar) {
    try {

      if (covar instanceof SvdMatrix) {

        final SvdMatrix svdCovar = (SvdMatrix) covar;
        final double largestEigenval =
            svdCovar.getSvd().getS().getElement(0, 0);
        final double varDistance = 3d * Math.sqrt(largestEigenval);
        return varDistance;
      } else {
        final no.uib.cipr.matrix.Matrix covarMtj =
            DenseMatrixFactoryMTJ.INSTANCE.copyMatrix(covar)
                .getInternalMatrix();
        final SymmDenseEVD evd =
            new SymmDenseEVD(covarMtj.numRows(), true, false)
                .factor(new UpperSymmDenseMatrix(covarMtj));
        final Double largestEigenval =
            Iterables.getLast(Doubles.asList(evd.getEigenvalues()));
        final double varDistance = 3d * Math.sqrt(largestEigenval);
        return varDistance;
      }
    } catch (final NotConvergedException e) {
      return Double.NaN;
    }
  }
 

}
