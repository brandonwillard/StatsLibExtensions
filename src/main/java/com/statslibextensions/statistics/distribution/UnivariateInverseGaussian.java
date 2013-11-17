package com.statslibextensions.statistics.distribution;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.google.common.collect.Lists;

import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.AbstractClosedFormSmoothUnivariateDistribution;
import gov.sandia.cognition.statistics.AbstractClosedFormUnivariateDistribution;
import gov.sandia.cognition.statistics.DistributionEstimator;
import gov.sandia.cognition.statistics.EstimableDistribution;
import gov.sandia.cognition.statistics.SmoothCumulativeDistributionFunction;
import gov.sandia.cognition.statistics.UnivariateProbabilityDensityFunction;

public class UnivariateInverseGaussian extends
    AbstractClosedFormSmoothUnivariateDistribution implements
    EstimableDistribution<Double, UnivariateInverseGaussian> {

  public class CDF extends UnivariateInverseGaussian implements
      SmoothCumulativeDistributionFunction {

    public CDF(UnivariateInverseGaussian other) {
      super(other.mu, other.lambda);
    }

    @Override
    public Double evaluate(Double input) {
      return null;
    }

    @Override
    public double evaluate(double input) {
      return Double.NaN;
    }

    @Override
    public double evaluateAsDouble(Double input) {
      return Double.NaN;
    }

    @Override
    public Double differentiate(Double input) {
      return null;
    }

    @Override
    public UnivariateProbabilityDensityFunction getDerivative() {
      return null;
    }

  }

  public class PDF extends UnivariateInverseGaussian
    implements UnivariateProbabilityDensityFunction {

    public PDF(double mu, double lambda) {
      super(mu, lambda);
    }

    public PDF(UnivariateInverseGaussian other) {
      super(other);
    }

    @Override
    public double logEvaluate(double input) {
      final double normCoef = (this.lambda/(2d * Math.PI * Math.pow(input, 3)))/2d;
      final double kernel = - this.lambda * Math.pow(input - this.mu, 2d)/
          (2d * this.mu * this.mu * input);
      return normCoef + kernel;
    }

    @Override
    public double logEvaluate(Double input) {
      return this.logEvaluate(input);
    }

    @Override
    public Double evaluate(Double input) {
      return this.evaluate(input.doubleValue());
    }

    @Override
    public double evaluate(double input) {
      return Math.exp(this.logEvaluate(input));
    }

    @Override
    public double evaluateAsDouble(Double input) {
      return this.evaluate(input);
    }

  }

  protected double mu;
  protected double lambda;

  public UnivariateInverseGaussian(double mu, double lambda) {
    this.mu = mu;
    this.lambda = lambda;
  }

  public UnivariateInverseGaussian(UnivariateInverseGaussian other) {
    this(other.mu, other.lambda);
  }

  @Override
  public AbstractClosedFormUnivariateDistribution<Double> clone() {
    UnivariateInverseGaussian clone = (UnivariateInverseGaussian) super.clone();
    clone.lambda = this.lambda;
    clone.mu = this.mu;
    return clone;
  }

  @Override
  public UnivariateInverseGaussian.PDF getProbabilityFunction() {
    return new UnivariateInverseGaussian.PDF(this);
  }

  @Override
  public UnivariateInverseGaussian.CDF getCDF() {
    return new UnivariateInverseGaussian.CDF(this);
  }

  @Override
  public Double getMean() {
    return this.mu;
  }
  
  @Override
  public Double sample(Random random) {
    return sample(this.mu, this.lambda, random);
  }

  @Override
  public ArrayList<Double> sample(Random random, int numSamples) {
    List<Double> samples = Lists.newArrayList();
    for (int i = 0; i < numSamples; i++) {
      samples.add(this.sample(random));
    }
    return (ArrayList<Double>) samples;
  }

  @Override
  public Vector convertToVector() {
    throw new UnsupportedOperationException();
  }

  @Override
  public void convertFromVector(Vector parameters) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Double getMinSupport() {
    return 0d;
  }

  @Override
  public Double getMaxSupport() {
    return Double.POSITIVE_INFINITY;
  }

  @Override
  public double getVariance() {
    return Math.pow(this.mu, 3) / this.lambda;
  }

  @Override
  public DistributionEstimator<Double, UnivariateInverseGaussian> getEstimator() {
    throw new UnsupportedOperationException();
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    long temp;
    temp = Double.doubleToLongBits(this.lambda);
    result = prime * result + (int) (temp ^ (temp >>> 32));
    temp = Double.doubleToLongBits(this.mu);
    result = prime * result + (int) (temp ^ (temp >>> 32));
    return result;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (!(obj instanceof UnivariateInverseGaussian)) {
      return false;
    }
    UnivariateInverseGaussian other = (UnivariateInverseGaussian) obj;
    if (Double.doubleToLongBits(this.lambda) != Double
        .doubleToLongBits(other.lambda)) {
      return false;
    }
    if (Double.doubleToLongBits(this.mu) != Double.doubleToLongBits(other.mu)) {
      return false;
    }
    return true;
  }

  public static double sample(double mu, double lambda, Random random) {
    final double v = random.nextGaussian(); 
    final double y = v*v;
    final double mu2 = mu * mu;
    final double mu2y = mu2 * y;
    final double lambdaT2 = 2d * lambda;
    final double x = mu + mu2y/lambdaT2 
        - mu/lambdaT2 * Math.sqrt(4d*mu*lambda*y + mu2y*y);
    final double test = random.nextDouble();  
    if (test <= mu/(mu + x))
      return x;
    else
      return mu2/x;
  }

}
