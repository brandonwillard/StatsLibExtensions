package com.statslibextensions.util;

import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.matrix.mtj.DenseMatrix;
import gov.sandia.cognition.math.matrix.mtj.decomposition.CholeskyDecompositionMTJ;
import gov.sandia.cognition.statistics.distribution.InverseWishartDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;

import java.util.Random;

import org.junit.Assert;
import org.junit.Test;

public class ExtStatisticsUtilTest {

  @Test
  public void testCovSqrt() {
    final Matrix mat =
        MatrixFactory.getDefault().copyArray(
            new double[][] { { 0.0001d, 0.00009d },
                { 0.00009d, 0.0001d } });

    final Matrix chol =
        CholeskyDecompositionMTJ.create((DenseMatrix) mat).getR();
    final Matrix covSqrt = ExtMatrixUtils.rootOfSemiDefinite(mat);
    final Matrix localChol = ExtMatrixUtils.getCholR(mat);

    Assert.assertTrue(chol.equals(localChol, 1e-5));

    final Random rng = new Random(1234567);

    MultivariateGaussian.sample(
        VectorFactory.getDefault().copyArray(new double[] { 0, 0 }),
        covSqrt, rng);

    rng.setSeed(1234567);

    MultivariateGaussian.sample(
        VectorFactory.getDefault().copyArray(new double[] { 0, 0 }),
        chol, rng);

    final MultivariateGaussian.SufficientStatistic ss1 =
        new MultivariateGaussian.SufficientStatistic();

    final MultivariateGaussian.SufficientStatistic ss2 =
        new MultivariateGaussian.SufficientStatistic();

    for (int i = 0; i < 500000; i++) {
      final Vector localSmpl1 =
          MultivariateGaussian.sample(VectorFactory.getDefault()
              .copyArray(new double[] { 0, 0 }), covSqrt, rng);
      final Vector localSmpl2 =
          MultivariateGaussian.sample(VectorFactory.getDefault()
              .copyArray(new double[] { 0, 0 }), chol, rng);
      ss1.update(localSmpl1);
      ss2.update(localSmpl2);
    }

    final Matrix cov1 = ss1.getCovariance();
    ss2.getCovariance();

    Assert.assertTrue(ss1.getMean().isZero(1e-5));
    Assert.assertTrue(mat.equals(cov1, 1e-5));
  }

  
  /**
   * Make sure the normal cdf works for the general, extreme limits.
   */
  @Test
  public void testNormalCdf() {

    for (int i = -1; i < 1; i++) {
      final double result1 = ExtStatisticsUtils.normalCdf(
          Double.POSITIVE_INFINITY, i*1d, 1d, false);
      Assert.assertEquals(1d, result1, 0d);

      final double result2 = ExtStatisticsUtils.normalCdf(
          Double.NEGATIVE_INFINITY, i*1d, 1d, false);
      Assert.assertEquals(0d, result2, 0d);
    }

    // Now, in log scale
    for (int i = -1; i < 1; i++) {
      final double result1 = ExtStatisticsUtils.normalCdf(
          Double.POSITIVE_INFINITY, i*1d, 1d, true);
      Assert.assertEquals(0d, result1, 0d);

      final double result2 = ExtStatisticsUtils.normalCdf(
          Double.NEGATIVE_INFINITY, i*1d, 1d, true);
      Assert.assertEquals(Double.NEGATIVE_INFINITY, result2, 0d);
    }
    
  }

}
