package com.statslibextensions.statistics.bayesian;

import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.signals.LinearDynamicalSystem;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;

import org.junit.Assert;
import org.junit.Test;

public class DlmUtilsTest {

  /**
   * Simple 1d test.
   */
//  @Test
//  public void testSvdForwardFilter1() {
//
//    Matrix modelCovariance1 = MatrixFactory.getDefault().copyArray(
//        new double[][] {{0d}});
//
//    Matrix measurementCovariance = MatrixFactory.getDefault().copyArray(
//        new double[][] {{1d}});
//
//    LinearDynamicalSystem model1 = new LinearDynamicalSystem(
//        MatrixFactory.getDefault().copyArray(new double[][] {{1d}}),
//        MatrixFactory.getDefault().copyArray(new double[][] {{0d}}),
//        MatrixFactory.getDefault().copyArray(new double[][] {{1d}})
//      );
//
//    KalmanFilter kf = new KalmanFilter(model1, modelCovariance1, measurementCovariance);
//    
//    MultivariateGaussian belief = kf.createInitialLearnedObject();
//    belief.setCovariance(MatrixFactory.getDefault().copyArray(new double[][] {{1}}));
//    
//    Vector obs1 = VectorFactory.getDefault().copyArray(new double[] {1d});
//    
//    MultivariateGaussian filtBelief2 = belief.clone();
//    DlmUtils.svdForwardFilter(obs1, filtBelief2, kf);
//    
//    
//    Assert.assertArrayEquals(new double[] {1d/2d}, 
//        filtBelief2.getMean().toArray(), 1e-7d);
//    Assert.assertArrayEquals(new double[] {1d/2d}, 
//        filtBelief2.getCovariance().convertToVector().toArray(), 1e-7d);
//  }

  /**
   * Simple 1d test.
   */
  @Test
  public void testSchurForwardFilter1() {

    Matrix modelCovariance1 = MatrixFactory.getDefault().copyArray(
        new double[][] {{0d}});

    Matrix measurementCovariance = MatrixFactory.getDefault().copyArray(
        new double[][] {{1d}});

    LinearDynamicalSystem model1 = new LinearDynamicalSystem(
        MatrixFactory.getDefault().copyArray(new double[][] {{1d}}),
        MatrixFactory.getDefault().copyArray(new double[][] {{0d}}),
        MatrixFactory.getDefault().copyArray(new double[][] {{1d}})
      );

    KalmanFilter kf = new KalmanFilter(model1, modelCovariance1, measurementCovariance);
    
    MultivariateGaussian belief = kf.createInitialLearnedObject();
    belief.setCovariance(MatrixFactory.getDefault().copyArray(new double[][] {{1}}));
    
    Vector obs1 = VectorFactory.getDefault().copyArray(new double[] {1d});
    
    MultivariateGaussian filtBelief2 = belief.clone();
    DlmUtils.schurForwardFilter(obs1, filtBelief2, kf);
    
    
    Assert.assertArrayEquals(new double[] {1d/2d}, 
        filtBelief2.getMean().toArray(), 1e-7d);
    Assert.assertArrayEquals(new double[] {1d/2d}, 
        filtBelief2.getCovariance().convertToVector().toArray(), 1e-7d);
    

  }

  /**
   * Simple 1d test.
   */
  @Test
  public void testSchurBackwardFilter1() {

    Matrix modelCovariance1 = MatrixFactory.getDefault().copyArray(
        new double[][] {{1d}});

    Matrix measurementCovariance = MatrixFactory.getDefault().copyArray(
        new double[][] {{1d}});

    LinearDynamicalSystem model1 = new LinearDynamicalSystem(
        MatrixFactory.getDefault().copyArray(new double[][] {{1d}}),
        MatrixFactory.getDefault().copyArray(new double[][] {{0d}}),
        MatrixFactory.getDefault().copyArray(new double[][] {{1d}})
      );

    KalmanFilter kf = new KalmanFilter(model1, modelCovariance1, measurementCovariance);
    
    MultivariateGaussian belief = kf.createInitialLearnedObject();
    belief.setMean(VectorFactory.getDefault().copyArray(new double[] {1d}));
    belief.setCovariance(MatrixFactory.getDefault().copyArray(new double[][] {{1d}}));
    
    Vector obs1 = VectorFactory.getDefault().copyArray(new double[] {0.5d});
    
    MultivariateGaussian filtBelief2 = belief.clone();
    DlmUtils.schurBackwardFilter(obs1, filtBelief2, kf);
    
    
    /*
     * FIXME TODO this isn't really much of a test, so create a real one.
     * Basically, it's checking that the smoothed result is between
     * the prior and the observed value.
     */
    Assert.assertTrue(filtBelief2.getMean().getElement(0) > 0.5);
    Assert.assertTrue(filtBelief2.getMean().getElement(0) < 1d);
    Assert.assertTrue(filtBelief2.getCovariance().getElement(0,0) > 0d);
    Assert.assertTrue(filtBelief2.getCovariance().getElement(0,0) < 1d);

//    Assert.assertArrayEquals(new double[] {1d/2d}, 
//        filtBelief2.getMean().toArray(), 1e-7d);
//    Assert.assertArrayEquals(new double[] {1d/2d}, 
//        filtBelief2.getCovariance().convertToVector().toArray(), 1e-7d);

  }

}
