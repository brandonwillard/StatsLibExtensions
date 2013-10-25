package com.statslibextensions.statistics.bayesian;

import gov.sandia.cognition.learning.algorithm.AbstractBatchAndIncrementalLearner;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.decomposition.AbstractSingularValueDecomposition;
import gov.sandia.cognition.math.matrix.mtj.decomposition.SingularValueDecompositionMTJ;
import gov.sandia.cognition.math.signals.LinearDynamicalSystem;
import gov.sandia.cognition.statistics.bayesian.RecursiveBayesianEstimator;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.util.ObjectUtil;

import com.google.common.base.Preconditions;
import com.statslibextensions.math.matrix.SvdMatrix;
import com.statslibextensions.math.matrix.decomposition.SimpleSingularValueDecomposition;
import com.statslibextensions.statistics.distribution.SvdMultivariateGaussian;
import com.statslibextensions.util.ExtMatrixUtils;

/**
 * An SVD-based Kalman Filter.
 * @see{}
 * 
 * @author bwillard
 */
public class SvdKalmanFilter
    extends
    AbstractBatchAndIncrementalLearner<Vector, SvdMultivariateGaussian>
    implements
    RecursiveBayesianEstimator<Vector, Vector, SvdMultivariateGaussian> {

  /**
   * Default autonomous dimension, {@value} .
   */
  public static final int DEFAULT_DIMENSION = 1;

  protected SvdMatrix measurementCovariance;

  /**
   * Motion model of the underlying system.
   */
  protected LinearDynamicalSystem model;
  protected SvdMatrix modelCovariance;

  /**
   * Creates a new instance of LinearUpdater
   * 
   * @param model
   *          Motion model of the underlying system.
   * @param modelCovariance
   *          Covariance associated with the system's model.
   * @param measurementCovariance
   *          Covariance associated with the measurements.
   */
  public SvdKalmanFilter(LinearDynamicalSystem model,
    SvdMatrix modelCovariance, SvdMatrix measurementCovariance) {
    this.measurementCovariance = measurementCovariance;
    this.modelCovariance = modelCovariance;
    this.setModel(model);
  }

  @Override
  public SvdKalmanFilter clone() {
    final SvdKalmanFilter clone =
        (SvdKalmanFilter) super.clone();
    clone.model = this.model.clone();
    clone.measurementCovariance =
        ObjectUtil.cloneSmart(this.measurementCovariance);
    clone.modelCovariance =
        ObjectUtil.cloneSmart(this.modelCovariance);
    return clone;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (this.getClass() != obj.getClass()) {
      return false;
    }
    final SvdKalmanFilter other =
        (SvdKalmanFilter) obj;
    if (this.model == null) {
      if (other.model != null) {
        return false;
      }
    } else if (!this.model.convertToVector().equals(other.model.convertToVector())) {
      return false;
    }
    return true;
  }

  public SvdMatrix getMeasurementCovariance() {
    return this.measurementCovariance;
  }

  /**
   * Getter for model
   * 
   * @return Motion model of the underlying system.
   */
  public LinearDynamicalSystem getModel() {
    return this.model;
  }

  public SvdMatrix getModelCovariance() {
    return this.modelCovariance;
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = super.hashCode();
    result =
        prime
            * result
            + ((this.model == null) ? 0 : this.model.convertToVector().hashCode());
    return result;
  }

  public void
      measure(MultivariateGaussian belief, Vector observation) {

    Preconditions.checkArgument(ExtMatrixUtils
        .isPosSemiDefinite(belief.getCovariance()));
    final Matrix F = this.model.getC();

    AbstractSingularValueDecomposition svdR;
    if (belief instanceof SvdMultivariateGaussian) {
      svdR =
          ((SvdMultivariateGaussian) belief).getCovariance().getSvd();
    } else {
      svdR =
          SingularValueDecompositionMTJ
              .create(belief.getCovariance());
    }

    final Matrix NvInv =
        ExtMatrixUtils.getDiagonalInverse(
            ExtMatrixUtils.getDiagonalSqrt(this.measurementCovariance
                .getSvd().getS(), 1e-7), 1e-7).times(
            this.measurementCovariance.getSvd().getU().transpose());
    final Matrix NvFU = NvInv.times(F).times(svdR.getU());
    final Matrix SRinv =
        ExtMatrixUtils.getDiagonalInverse(
            ExtMatrixUtils.getDiagonalSqrt(svdR.getS(), 1e-7), 1e-7);
    final int nN2 = NvFU.getNumRows() + SRinv.getNumRows();
    final int nM2 = SRinv.getNumColumns();
    final Matrix M2 =
        MatrixFactory.getDefault().createMatrix(nN2, nM2);
    M2.setSubMatrix(0, 0, NvFU);
    M2.setSubMatrix(NvFU.getNumRows(), 0, SRinv);

    final AbstractSingularValueDecomposition svdM2 =
        SingularValueDecompositionMTJ.create(M2);
    final Matrix S =
        MatrixFactory.getDefault().createMatrix(
            svdM2.getS().getNumColumns(),
            svdM2.getS().getNumColumns());
    for (int i = 0; i < Math.min(svdM2.getS().getNumColumns(), svdM2
        .getS().getNumRows()); i++) {
      final double sVal = svdM2.getS().getElement(i, i);
      final double sValInvSq = 1d / (sVal * sVal);
      if (sValInvSq > 1e-7) {
        S.setElement(i, i, sValInvSq);
      }
    }
    final Matrix UcNew =
        svdR.getU().times(svdM2.getVtranspose().transpose());
    final AbstractSingularValueDecomposition svdCnew =
        new SimpleSingularValueDecomposition(UcNew, S,
            UcNew.transpose());

    Preconditions.checkArgument(ExtMatrixUtils
        .isPosSemiDefinite(UcNew.times(S).times(UcNew.transpose())));

    final SvdMatrix Q =
        ExtMatrixUtils.symmetricSvdAdd(
            (SvdMatrix) belief.getCovariance(),
            this.measurementCovariance, F);
    final Matrix Qinv =
        Q.getSvd()
            .getU()
            .times(
                ExtMatrixUtils.getDiagonalInverse(Q.getSvd().getS(),
                    1e-7)).times(Q.getSvd().getU().transpose());
    Preconditions.checkArgument(ExtMatrixUtils
        .isPosSemiDefinite(Qinv));
    final Vector e = observation.minus(F.times(belief.getMean()));

    final Matrix A =
        belief.getCovariance().times(F.transpose()).times(Qinv);

    final Vector postMean = belief.getMean().plus(A.times(e));


    belief.setMean(postMean);
    if (belief instanceof SvdMultivariateGaussian) {
      ((SvdMultivariateGaussian) belief).getCovariance().setSvd(
          svdCnew);
    } else {
      belief.setCovariance(svdCnew.getU().times(svdCnew.getS())
          .times(svdCnew.getVtranspose()));
    }
  }

  public void predict(MultivariateGaussian belief) {

    final Matrix G = this.model.getA();
    AbstractSingularValueDecomposition svdC;
    if (belief instanceof SvdMultivariateGaussian) {
      svdC =
          ((SvdMultivariateGaussian) belief).getCovariance().getSvd();
    } else {
      svdC =
          SingularValueDecompositionMTJ
              .create(belief.getCovariance());
    }
    final Matrix SUG =
        ExtMatrixUtils.getDiagonalSqrt(svdC.getS(), 1e-7)
            .times(svdC.getU().transpose()).times(G.transpose());
    final Matrix Nw =
        ExtMatrixUtils.getDiagonalSqrt(
            this.modelCovariance.getSvd().getS(), 1e-7).times(
            this.modelCovariance.getSvd().getU().transpose());
    final int nN = SUG.getNumRows() + Nw.getNumRows();
    final int nM = SUG.getNumColumns();
    final Matrix M1 = MatrixFactory.getDefault().createMatrix(nN, nM);
    M1.setSubMatrix(0, 0, SUG);
    M1.setSubMatrix(SUG.getNumRows(), 0, Nw);

    final AbstractSingularValueDecomposition svdM =
        SingularValueDecompositionMTJ.create(M1);
    final Matrix S = ExtMatrixUtils.getDiagonalSquare(svdM.getS(), 1e-7);

    final AbstractSingularValueDecomposition svdR =
        new SimpleSingularValueDecomposition(svdM.getVtranspose()
            .transpose(), S, svdM.getVtranspose());

    final Matrix R;
    if (belief instanceof SvdMultivariateGaussian) {
      R = new SvdMatrix(svdR);
    } else {
      R = svdR.getU().times(svdR.getS()).times(svdR.getVtranspose());
    }
    /*
     * Check that we maintain numerical accuracy for our given model
     * design (in which the state covariances are always degenerate).
     */
    //    Preconditions.checkState((belief.getInputDimensionality() != 2 || svdR.rank() == 1)
    //        && (belief.getInputDimensionality() != 4 || svdR.rank() == 2));
    //    Preconditions.checkState(svdR.getU().getNumRows() == 2
    //        || svdR.getU().getNumRows() == 4);

    belief.setMean(G.times(belief.getMean()));
    belief.setCovariance(R);

    Preconditions.checkState(belief.getCovariance().isSquare()
        && belief.getCovariance().isSymmetric());
  }

  public void
      setMeasurementCovariance(SvdMatrix measurementCovariance) {
    this.measurementCovariance = measurementCovariance;
  }

  /**
   * Setter for model
   * 
   * @param model
   *          Motion model of the underlying system.
   */
  public void setModel(LinearDynamicalSystem model) {
    this.model = model;
  }

  public void setModelCovariance(SvdMatrix modelCovariance) {
    this.modelCovariance = modelCovariance;
  }

  @Override
  public void update(SvdMultivariateGaussian target, Vector data) {
    this.measure(target, data);
  }

	@Override
	public SvdMultivariateGaussian createInitialLearnedObject() {
	    return new SvdMultivariateGaussian(
	        this.model.getState(), this.getModelCovariance() );
	}

}
