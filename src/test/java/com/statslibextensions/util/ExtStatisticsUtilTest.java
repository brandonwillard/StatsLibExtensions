package com.statslibextensions.util;

import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.matrix.mtj.DenseMatrix;
import gov.sandia.cognition.math.matrix.mtj.decomposition.CholeskyDecompositionMTJ;
import gov.sandia.cognition.statistics.distribution.InverseWishartDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;

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

  @Test
  public void testInvWishartSampling() {
    final Random rng = //new Random();
        new Random(123456789);

    final Matrix mean =
        MatrixFactory.getDefault().copyArray(
            new double[][] { { 100d, 0d }, { 0d, 100d } });
    final InverseWishartDistribution invWish =
        new InverseWishartDistribution(MatrixFactory.getDefault()
            .copyArray(
                new double[][] { { 1700d, 0d }, { 0d, 1700d } }), 20);

    MultivariateGaussian.SufficientStatistic ss =
        new MultivariateGaussian.SufficientStatistic();
    for (int j = 0; j < 10; j++) {
      ss = new MultivariateGaussian.SufficientStatistic();
      for (int i = 0; i < 10000; i++) {
        final Matrix smpl2 =
            ExtStatisticsUtils.sampleInvWishart(invWish, rng);
        ss.update(mean.minus(smpl2).convertToVector());
      }
      System.out.println(ss.getMean());
    }

    Assert.assertTrue(Math.abs(ss.getMean().sum()) < 1d);

  }

}
