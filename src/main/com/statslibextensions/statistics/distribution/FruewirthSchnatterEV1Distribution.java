package com.statslibextensions.statistics.distribution;

import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.ClosedFormComputableDistribution;
import gov.sandia.cognition.statistics.ProbabilityDensityFunction;
import gov.sandia.cognition.statistics.ProbabilityFunction;
import gov.sandia.cognition.statistics.distribution.LinearMixtureModel;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.google.common.collect.ImmutableList;

/**
 * Mixture model approximation to a type I extreme value distribution. <br>
 * From 
 * S. Fruewirth-Schnatter, R. Fruewirth "Auxiliary mixture sampling with applications to logistic models" 
 * 
 * @author bwillard
 * 
 */
public class FruewirthSchnatterEV1Distribution extends
    LinearMixtureModel<Double, UnivariateGaussian> implements
    ClosedFormComputableDistribution<Double> {

  final protected static List<UnivariateGaussian> fsEV1distributions = ImmutableList
      .<UnivariateGaussian> builder().add(new UnivariateGaussian(5.09d, 4.50d))
      .add(new UnivariateGaussian(3.29d, 2.02d))
      .add(new UnivariateGaussian(1.82d, 1.10d))
      .add(new UnivariateGaussian(1.24d, 0.422d))
      .add(new UnivariateGaussian(0.746d, 0.198d))
      .add(new UnivariateGaussian(0.391d, 0.107d))
      .add(new UnivariateGaussian(0.0431d, 0.0778d))
      .add(new UnivariateGaussian(-0.306d, 0.0766d))
      .add(new UnivariateGaussian(-0.673d, 0.0947d))
      .add(new UnivariateGaussian(-1.06d, 0.146d)).build();

  final protected static double[] fsEV1priorWeights = new double[] { 0.00397d,
      0.0396d, 0.168d, 0.147d, 0.125d, 0.101d, 0.104d, 0.116d, 0.107d, 0.088d };

  public FruewirthSchnatterEV1Distribution() {
    super(fsEV1distributions, fsEV1priorWeights);
  }

  /**
   * Copy Constructor
   * 
   * @param other
   *          MultivariateMixtureDensityModel to copy
   */
  public FruewirthSchnatterEV1Distribution(
      FruewirthSchnatterEV1Distribution other) {
    super(fsEV1distributions, fsEV1priorWeights);
  }

  @Override
  public FruewirthSchnatterEV1Distribution clone() {
    return this;
  }

  @Override
  public void convertFromVector(Vector parameters) {
  }

  protected static FruewirthSchnatterEV1Distribution.PDF fsPDF = new FruewirthSchnatterEV1Distribution.PDF();

  @Override
  public FruewirthSchnatterEV1Distribution.PDF getProbabilityFunction() {
    return fsPDF;
  }

  @Override
  public String toString() {
    final StringBuilder retval = new StringBuilder(1000);
    retval.append("FruewirthSchnatter EV1 approx.:\n");
    int k = 0;
    for (final UnivariateGaussian distribution : this.getDistributions()) {
      retval.append(" " + k + ": Prior: " + this.getPriorWeights()[k]
          + ", Distribution:\nMean: " + distribution.getMean() + "\nVariance:"
          + distribution.getVariance() + "\n");
      k++;
    }
    return retval.toString();
  }

  @Override
  public Double getMean() {
    return 1.05811d;
  }

  @Override
  public Vector convertToVector() {
    return VectorFactory.getDefault().copyArray(fsEV1priorWeights);
  }

  public static class PDF extends FruewirthSchnatterEV1Distribution implements
      ProbabilityDensityFunction<Double> {

    protected PDF() {
      super();
    }

    @Override
    public PDF getProbabilityFunction() {
      return this;
    }

    @Override
    public double logEvaluate(Double input) {
      return Math.log(this.evaluate(input));
    }

    @Override
    public Double evaluate(Double input) {
      double sum = 0.0;
      final int K = this.getDistributionCount();
      for (int k = 0; k < K; k++) {
        ProbabilityFunction<Double> pdf = this.getDistributions().get(k)
            .getProbabilityFunction();
        sum += pdf.evaluate(input) * this.priorWeights[k];
      }

      return sum / this.getPriorWeightSum();
    }

    public double[] computeRandomVariableProbabilities(Double input) {
      int K = this.getDistributionCount();
      double[] likelihoods = this.computeRandomVariableLikelihoods(input);
      double sum = 0.0;
      for (int k = 0; k < K; k++) {
        sum += likelihoods[k];
      }
      if (sum <= 0.0) {
        Arrays.fill(likelihoods, 1.0 / K);
      }

      sum = 0.0;
      for (int k = 0; k < K; k++) {
        likelihoods[k] *= this.priorWeights[k];
        sum += likelihoods[k];
      }
      if (sum <= 0.0) {
        Arrays.fill(likelihoods, 1.0 / K);
        sum = 1.0;
      }
      for (int k = 0; k < K; k++) {
        likelihoods[k] /= sum;
      }

      return likelihoods;

    }

    /**
     * Computes the likelihoods of the underlying distributions
     * 
     * @param input
     *          Input to consider
     * @return Vector of likelihoods for the underlying distributions
     */
    public double[] computeRandomVariableLikelihoods(Double input) {

      int K = this.getDistributionCount();
      double[] likelihoods = new double[K];
      for (int k = 0; k < K; k++) {
        ProbabilityFunction<Double> pdf = this.getDistributions().get(k)
            .getProbabilityFunction();
        likelihoods[k] = pdf.evaluate(input);
      }

      return likelihoods;
    }

    /**
     * Gets the index of the most-likely distribution, given the input. That is,
     * find the distribution that most likely generated the input
     * 
     * @param input
     *          input to consider
     * @return zero-based index of the most-likely distribution
     */
    public int getMostLikelyRandomVariable(Double input) {

      double[] probabilities = this.computeRandomVariableProbabilities(input);
      int bestIndex = 0;
      double bestProbability = probabilities[0];
      for (int i = 1; i < probabilities.length; i++) {
        double prob = probabilities[i];
        if (bestProbability < prob) {
          bestProbability = prob;
          bestIndex = i;
        }
      }

      return bestIndex;

    }

    @Override
    public void setDistributions(
        ArrayList<? extends UnivariateGaussian> distributions) {
      // FIXME these values are constant; we shouldn't use this interface, i suppose.
      throw new IllegalAccessError();
    }

    @Override
    public void setPriorWeights(double[] priorWeights) {
      // FIXME these values are constant; we shouldn't use this interface, i suppose.
      throw new IllegalAccessError();
    }

    @Override
    public double getPriorWeightSum() {
      return 10.5811d;
    }

  }

}
