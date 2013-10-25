package com.statslibextensions.statistics.bayesian;

import com.statslibextensions.util.ExtMatrixUtils;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.UpperSPDDenseMatrix;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.matrix.mtj.AbstractMTJMatrix;
import gov.sandia.cognition.math.matrix.mtj.DenseMatrixFactoryMTJ;
import gov.sandia.cognition.math.matrix.mtj.DenseVectorFactoryMTJ;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;

public class DlmUtils {

  /**
   * Smoothing step of the given kalman filter for the given prior belief
   * and current observation. 
   * I.e. transforms {@code belief} from p(x<sub>t</sub>|y<sub>t</sub>) 
   * to p(x<sub>t-1</sub>|y<sub>t</sub>)
   * 
   * @param observation
   * @param belief
   * @param kalmanFilter
   */
  public static void backwardFilter(Vector observation, 
      MultivariateGaussian belief, KalmanFilter kalmanFilter) {
    Matrix F = kalmanFilter.getModel().getC();
    Matrix G = kalmanFilter.getModel().getA();
    Matrix C = belief.getCovariance();
    Vector m = belief.getMean();
    Matrix Omega = kalmanFilter.getModelCovariance();
    Matrix Sigma = kalmanFilter.getMeasurementCovariance();

    final Matrix W = F.times(Omega).times(F.transpose()).plus(Sigma);
    final Matrix FG = F.times(G);
    final Matrix A = FG.times(C).times(FG.transpose()).plus(W);
    /*
     * TODO: make solver configurable/conditional (e.g. on matrix size)
     */
    final Matrix Wtil =
        A.transpose().solve(FG.times(C.transpose())).transpose();

    final Vector mSmooth = m.plus(Wtil.times(observation.minus(FG.times(m))));
    final Matrix CSmooth =
        C.minus(Wtil.times(A).times(Wtil.transpose()));

    belief.setMean(mSmooth);
    belief.setCovariance(CSmooth);
  }

  public static void mtjForwardFilter(Vector observation, 
      MultivariateGaussian belief, KalmanFilter kalmanFilter) {

    final Matrix F = kalmanFilter.getModel().getC();
    final Vector a = belief.getMean();
    final Matrix R = belief.getCovariance();
    final Matrix Q =
        F.times(R).times(F.transpose())
            .plus(kalmanFilter.getMeasurementCovariance());
    /*
     * This is the source of one major improvement:
     * uses the solve routine for a positive definite matrix
     */
    final UpperSPDDenseMatrix Qspd =
        new UpperSPDDenseMatrix(
            ((AbstractMTJMatrix) Q).getInternalMatrix(), false);
    final no.uib.cipr.matrix.Matrix CRt =
        ((AbstractMTJMatrix) F.times(R.transpose()))
            .getInternalMatrix();

    final DenseMatrix Amtj =
        new DenseMatrix(Qspd.numRows(), CRt.numColumns());
    /*
     * TODO: make solver configurable/conditional (e.g. on matrix size)
     */
    Qspd.transSolve(CRt, Amtj);

    final DenseMatrix AtQt =
        new DenseMatrix(Amtj.numColumns(), Qspd.numRows());
    Amtj.transABmult(Qspd, AtQt);

    final DenseMatrix AtQtAMtj =
        new DenseMatrix(AtQt.numRows(), Amtj.numColumns());
    AtQt.mult(Amtj, AtQtAMtj);

    final Matrix AtQtA =
        ((DenseMatrixFactoryMTJ) MatrixFactory.getDenseDefault())
            .createWrapper(AtQtAMtj);

    final DenseVector e2 =
        new DenseVector(
            ((gov.sandia.cognition.math.matrix.mtj.DenseVector) observation
                .minus(F.times(a))).getArray(), false);

    final DenseVector AteMtj = new DenseVector(Amtj.numColumns());
    Amtj.transMult(e2, AteMtj);
    final Vector Ate =
        ((DenseVectorFactoryMTJ) VectorFactory.getDenseDefault())
            .createWrapper(AteMtj);

    final Matrix CC = R.minus(AtQtA);
    final Vector m = a.plus(Ate);

    assert ExtMatrixUtils 
        .isPosSemiDefinite((gov.sandia.cognition.math.matrix.mtj.DenseMatrix) CC);

    belief.setCovariance(CC);
    belief.setMean(m);
  }
}
