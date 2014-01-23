package com.statslibextensions.statistics.bayesian;

import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.matrix.mtj.AbstractMTJMatrix;
import gov.sandia.cognition.math.matrix.mtj.DenseMatrixFactoryMTJ;
import gov.sandia.cognition.math.matrix.mtj.DenseVectorFactoryMTJ;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;

import java.util.List;
import java.util.Random;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.UpperSPDDenseMatrix;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import com.statslibextensions.util.ExtMatrixUtils;
import com.statslibextensions.util.ObservedValue.SimObservedValue;

public class DlmUtils {
  
//  public static void svdPredict(MultivariateGaussian belief, 
//    KalmanFilter kalmanFilter) {
//    final Matrix G = kalmanFilter.getModel().getA();
//
//    final AbstractSingularValueDecomposition svdC;
//    if (belief instanceof SvdMultivariateGaussian) {
//      svdC =
//          ((SvdMultivariateGaussian) belief).getCovariance().getSvd();
//    } else {
//      svdC =
//          SingularValueDecompositionMTJ
//              .create(belief.getCovariance());
//    }
//
//    final AbstractSingularValueDecomposition svdModelCov; 
//    if (kalmanFilter.getModelCovariance() instanceof SvdMatrix) {
//      svdModelCov =
//          ((SvdMultivariateGaussian) kalmanFilter.getModelCovariance())
//          .getCovariance().getSvd();
//    } else {
//      svdModelCov =
//          SingularValueDecompositionMTJ
//              .create(kalmanFilter.getModelCovariance());
//    }
//    final Matrix SUG =
//        ExtMatrixUtils.getDiagonalSqrt(svdC.getS(), 1e-7)
//            .times(svdC.getU().transpose()).times(G.transpose());
//    final Matrix Nw =
//        ExtMatrixUtils.getDiagonalSqrt(
//            svdModelCov.getS(), 1e-7).times(
//            svdModelCov.getU().transpose());
//    final int nN = SUG.getNumRows() + Nw.getNumRows();
//    final int nM = SUG.getNumColumns();
//    final Matrix M1 = MatrixFactory.getDefault().createMatrix(nN, nM);
//    M1.setSubMatrix(0, 0, SUG);
//    M1.setSubMatrix(SUG.getNumRows(), 0, Nw);
//
//    final AbstractSingularValueDecomposition svdM =
//        SingularValueDecompositionMTJ.create(M1);
//    final Matrix S = ExtMatrixUtils.getDiagonalSquare(svdM.getS(), 1e-7);
//
//    final AbstractSingularValueDecomposition svdR =
//        new SimpleSingularValueDecomposition(svdM.getVtranspose()
//            .transpose(), S, svdM.getVtranspose());
//
//    final Matrix R;
//    if (belief instanceof SvdMultivariateGaussian) {
//      R = new SvdMatrix(svdR);
//    } else {
//      R = svdR.getU().times(svdR.getS()).times(svdR.getVtranspose());
//    }
//    /*
//     * Check that we maintain numerical accuracy for our given model
//     * design (in which the state covariances are always degenerate).
//     */
//    //    Preconditions.checkState((belief.getInputDimensionality() != 2 || svdR.rank() == 1)
//    //        && (belief.getInputDimensionality() != 4 || svdR.rank() == 2));
//    //    Preconditions.checkState(svdR.getU().getNumRows() == 2
//    //        || svdR.getU().getNumRows() == 4);
//
//    belief.setMean(G.times(belief.getMean()));
//    belief.setCovariance(R);
//
//    Preconditions.checkState(belief.getCovariance().isSquare()
//        && belief.getCovariance().isSymmetric());
//  }
//  
//  /**
//   * Forward filtering using SVDs.
//   * 
//   * XXX: Broken!
//   * TODO: fix this!
//   * 
//   * @param belief
//   * @param observation
//   */
//  public static void
//      svdForwardFilter(Vector observation, 
//      MultivariateGaussian belief, KalmanFilter kalmanFilter) {
//
//    Preconditions.checkArgument(ExtMatrixUtils
//        .isPosSemiDefinite(belief.getCovariance()));
//    final Matrix F = kalmanFilter.getModel().getC();
//
//    final SvdMatrix svdRMat;
//    final AbstractSingularValueDecomposition svdR;
//    if (belief instanceof SvdMultivariateGaussian) {
//      svdRMat = (SvdMatrix) belief.getCovariance();
//      svdR =
//          ((SvdMultivariateGaussian) belief).getCovariance().getSvd();
//    } else {
//      svdR =
//          SingularValueDecompositionMTJ
//              .create(belief.getCovariance());
//      svdRMat = new SvdMatrix(svdR);
//    }
//
//    final AbstractSingularValueDecomposition svdMeasureCov; 
//    if (kalmanFilter.getMeasurementCovariance() instanceof SvdMatrix) {
//      svdMeasureCov =
//          ((SvdMultivariateGaussian) kalmanFilter.getMeasurementCovariance())
//          .getCovariance().getSvd();
//    } else {
//      svdMeasureCov =
//          SingularValueDecompositionMTJ
//              .create(kalmanFilter.getMeasurementCovariance());
//    }
//
//    final AbstractSingularValueDecomposition svdModelCov; 
//    final SvdMatrix modelCov;
//    if (kalmanFilter.getModelCovariance() instanceof SvdMatrix) {
//      modelCov = ((SvdMultivariateGaussian) kalmanFilter.getModelCovariance())
//          .getCovariance();
//      svdModelCov =
//          ((SvdMultivariateGaussian) kalmanFilter.getModelCovariance())
//          .getCovariance().getSvd();
//    } else {
//      svdModelCov =
//          SingularValueDecompositionMTJ
//              .create(kalmanFilter.getModelCovariance());
//      modelCov = new SvdMatrix(svdModelCov);
//    }
//
//
//    final Matrix NvInv =
//        ExtMatrixUtils.getDiagonalInverse(
//            ExtMatrixUtils.getDiagonalSqrt(
//                svdMeasureCov.getS(), 1e-7), 1e-7).times(
//            svdMeasureCov.getU().transpose());
//    final Matrix NvFU = NvInv.times(F).times(svdR.getU());
//    final Matrix SRinv =
//        ExtMatrixUtils.getDiagonalInverse(
//            ExtMatrixUtils.getDiagonalSqrt(svdR.getS(), 1e-7), 1e-7);
//    final int nN2 = NvFU.getNumRows() + SRinv.getNumRows();
//    final int nM2 = SRinv.getNumColumns();
//    final Matrix M2 =
//        MatrixFactory.getDefault().createMatrix(nN2, nM2);
//    M2.setSubMatrix(0, 0, NvFU);
//    M2.setSubMatrix(NvFU.getNumRows(), 0, SRinv);
//
//    final AbstractSingularValueDecomposition svdM2 =
//        SingularValueDecompositionMTJ.create(M2);
//    final Matrix S =
//        MatrixFactory.getDefault().createMatrix(
//            svdM2.getS().getNumColumns(),
//            svdM2.getS().getNumColumns());
//    for (int i = 0; i < Math.min(svdM2.getS().getNumColumns(), svdM2
//        .getS().getNumRows()); i++) {
//      final double sVal = svdM2.getS().getElement(i, i);
//      final double sValInvSq = 1d / (sVal * sVal);
//      if (sValInvSq > 1e-7) {
//        S.setElement(i, i, sValInvSq);
//      }
//    }
//    final Matrix UcNew =
//        svdR.getU().times(svdM2.getVtranspose().transpose());
//    final AbstractSingularValueDecomposition svdCnew =
//        new SimpleSingularValueDecomposition(UcNew, S,
//            UcNew.transpose());
//
//    Preconditions.checkArgument(ExtMatrixUtils
//        .isPosSemiDefinite(UcNew.times(S).times(UcNew.transpose())));
//
//    final SvdMatrix Q =
//        ExtMatrixUtils.symmetricSvdAdd(
//            svdRMat, modelCov, F);
//    final Matrix Qinv =
//        Q.getSvd()
//            .getU()
//            .times(
//                ExtMatrixUtils.getDiagonalInverse(Q.getSvd().getS(),
//                    1e-7)).times(Q.getSvd().getU().transpose());
//    Preconditions.checkArgument(ExtMatrixUtils
//        .isPosSemiDefinite(Qinv));
//    final Vector e = observation.minus(F.times(belief.getMean()));
//
//    final Matrix A =
//        belief.getCovariance().times(F.transpose()).times(Qinv);
//    final Vector postMean = belief.getMean().plus(A.times(e));
//
//    belief.setMean(postMean);
//    if (belief instanceof SvdMultivariateGaussian) {
//      ((SvdMultivariateGaussian) belief).getCovariance().setSvd(
//          svdCnew);
//    } else {
//      belief.setCovariance(svdCnew.getU().times(svdCnew.getS())
//          .times(svdCnew.getVtranspose()));
//    }
//  }

  /**
   * Filter/Bayes step for {@code kalmanFilter}, the given prior {@code belief}
   * and the current {@code observation}. 
   * I.e. transforms {@code belief} from p(x<sub>t-1</sub>|y<sub>t</sub>) 
   * to p(x<sub>t</sub>|y<sub>t</sub>)
   * <br>
   * The method used is from a Schur complement approach, using * Matrix.solve
   * for the single inversion.
   * 
   * @see Matrix#solve(Vector)
   * @param observation
   * @param belief
   * @param kalmanFilter
   */
  public static void schurForwardFilter(Vector observation, 
      MultivariateGaussian belief, KalmanFilter kalmanFilter) {
    Matrix F = kalmanFilter.getModel().getC();
    Matrix G = kalmanFilter.getModel().getA();
    Matrix C = belief.getCovariance();
    Vector m = belief.getMean();
    Matrix Omega = kalmanFilter.getModelCovariance();
    Matrix Sigma = kalmanFilter.getMeasurementCovariance();

    /*
     * Since our prior is p(x_{t-1}|y_{t-1}), we need to
     * compute the prior predictives, i.e. p(x_t | y_{t-1}) and p(y_t | y_{t-1}),
     * here.
     */
    final Vector a = G.times(m);
    final Matrix R = G.times(C).times(G.transpose()).plus(Omega);
    final Vector f = F.times(a);
    final Matrix Q = F.times(R).times(F.transpose()).plus(Sigma);

    /*
     * Compute the Kalman gain...
     * TODO: make solver configurable/conditional (e.g. on matrix size)
     */
    final Matrix RFt = R.times(F.transpose());
    final Matrix K = Q.solve(RFt.transpose()).transpose();
    final Vector resid = observation.minus(f);

    final Vector mSmooth = a.plus(K.times(resid));
    final Matrix CSmooth = 
        R.minus(K.times(RFt.transpose()));

    belief.setMean(mSmooth);
    belief.setCovariance(CSmooth);
  }

  /**
   * Performs one smoothing step for {@code kalmanFilter}, 
   * the given prior {@code prior}
   * and the current {@code observation}. 
   * I.e. transforms {@code belief} from p(x<sub>t</sub>|y<sub>t</sub>) 
   * to p(x<sub>t-1</sub>|y<sub>t</sub>)
   * <br>
   * The method used is from a Schur complement approach, using * Matrix.solve
   * for two inversions.
   * 
   * @see Matrix#solve(Vector)
   * @param observation Observation at time {@literal t}
   * @param prior Prior distribution at time {@literal t-1}
   * @param kalmanFilter
   */
  public static void schurBackwardFilter(Vector observation, 
      MultivariateGaussian prior, KalmanFilter kalmanFilter) {
    Matrix F = kalmanFilter.getModel().getC();
    Matrix G = kalmanFilter.getModel().getA();
    Matrix C = prior.getCovariance();
    Vector m = prior.getMean();
    Matrix Omega = kalmanFilter.getModelCovariance();
    Matrix Sigma = kalmanFilter.getMeasurementCovariance();

    /*
     * Since our prior is p(x_{t-1}|y_{t-1}), we need to
     * compute the prior predictives, i.e. p(x_t | y_{t-1}) and p(y_t | y_{t-1}),
     * here.
     */
    final Vector a = G.times(m);
    final Matrix R = G.times(C).times(G.transpose()).plus(Omega);
    final Matrix RFt = R.times(F.transpose());
    final Matrix Q = F.times(RFt).plus(Sigma);
    /*
     * TODO: make solver configurable/conditional (e.g. on matrix size)
     */
    final Matrix JK =
        Q.solve(F.times(G).times(R)).transpose();
    final Matrix K = Q.solve(RFt.transpose()).transpose();


    final Vector resid = observation.minus(F.times(a));
    final Vector mSmooth = m.plus(JK.times(resid));
    final Matrix CSmooth = C.minus(K.times(F).times(C));

    prior.setMean(mSmooth);
    prior.setCovariance(CSmooth);
  }

  /**
   * Performs one smoothing step for {@code kalmanFilter}, 
   * the given {@code prior}
   * and {@code posterior}. 
   * I.e. transforms {@code belief} from p(x<sub>t</sub>|y<sub>t</sub>) 
   * to p(x<sub>t-1</sub>|y<sub>t</sub>)
   * <br>
   * The method used is from a Schur complement approach, using * Matrix.solve
   * for the single inversion.
   * 
   * @see Matrix#solve(Vector)
   * @param observation Observation at time {@literal t}
   * @param prior Prior distribution at time {@literal t-1}
   * @param kalmanFilter
   */
  public static void schurBackwardFilter(MultivariateGaussian posterior, 
      MultivariateGaussian prior, KalmanFilter kalmanFilter) {
    Matrix F = kalmanFilter.getModel().getC();
    Matrix G = kalmanFilter.getModel().getA();
    Matrix C_tm1 = prior.getCovariance();
    Vector m_tm1 = prior.getMean();
    Matrix Omega = kalmanFilter.getModelCovariance();
    Matrix Sigma = kalmanFilter.getMeasurementCovariance();

    final Vector a = G.times(m_tm1);
    final Matrix GC = G.times(C_tm1);
    final Matrix R_tm1 = GC.times(G.transpose()).plus(Omega);
    final Matrix RFt_tm1 = R_tm1.times(F.transpose());
    final Matrix Q_tm1 = F.times(RFt_tm1).plus(Sigma);
    /*
     * TODO: make solver configurable/conditional (e.g. on matrix size)
     */
    final Vector m_t = posterior.getMean();
    final Matrix C_t = posterior.getCovariance();
    final Matrix B = Q_tm1.solve(GC).transpose();


    final Vector resid = m_t.minus(a);
    final Vector mSmooth = m_t.plus(B.times(resid));
    final Matrix CSmooth = C_tm1.minus(B.times(C_t.minus(R_tm1)).times(B.transpose()));

    prior.setMean(mSmooth);
    prior.setCovariance(CSmooth);
  }

  /**
   * DLM forward filtering for MTJ objects.
   * 
   * @see KalmanFilter#measure(MultivariateGaussian, Vector)
   * 
   * @param observation
   * @param belief
   * @param kalmanFilter
   */
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
        .isPosSemiDefinite(CC);

    belief.setCovariance(CC);
    belief.setMean(m);
  }

//  public static MultivariateGaussian getSmoothedPostDist(MultivariateGaussian postBeta, 
//                                                   MultivariateGaussian augResponseDist, 
//                                                   ObservedValue<Vector, Matrix> observation, 
//                                                   Vector obsMeanAdj) {
//    final Matrix C = postBeta.getCovariance();
//    final Vector m = postBeta.getMean();
//    
//    // System design
//    final Matrix F = observation.getObservedData();
//    final Matrix G = MatrixFactory.getDefault().createIdentity(m.getDimensionality(), m.getDimensionality());
//    final Matrix Omega = MatrixFactory.getDefault().createIdentity(m.getDimensionality(), m.getDimensionality()); 
//    
//    // Observation suff. stats 
//    final Matrix Sigma = augResponseDist.getCovariance();
//    final Vector y = augResponseDist.getMean().minus(obsMeanAdj);
//    
//    final Vector a = G.times(m);
//    final Matrix R = Omega;
//    
//    final Matrix W = F.times(Omega).times(F.transpose()).plus(Sigma);
//    final Matrix FG = F.times(G);
//    final Matrix A = FG.times(R).times(FG.transpose()).plus(W);
//    final Matrix Wtil =
//        A.transpose().solve(FG.times(R.transpose())).transpose();
//
//    final Vector aSmooth = a.plus(Wtil.times(y.minus(FG.times(a))));
//    final Matrix RSmooth =
//        R.minus(Wtil.times(A).times(Wtil.transpose()));
//    
//    return new MultivariateGaussian(aSmooth, RSmooth);
//  }

//  public static MultivariateGaussian getSmoothedPriorDist(MultivariateGaussian priorBeta, 
//                                                    MultivariateGaussian augResponseDist, 
//                                                    ObservedValue<Vector, Matrix> observation, Vector obsMeanAdj) {
//    // Prior suff. stats 
//    final Matrix C = priorBeta.getCovariance();
//    final Vector m = priorBeta.getMean();
//    
//    // System design
//    final Matrix F = observation.getObservedData();
//    final Matrix G = MatrixFactory.getDefault().createIdentity(m.getDimensionality(), m.getDimensionality());
//    final Matrix Omega = MatrixFactory.getDefault().createIdentity(m.getDimensionality(), m.getDimensionality()); 
//    
//    // Observation suff. stats 
//    final Matrix Sigma = augResponseDist.getCovariance();
//    final Vector y = augResponseDist.getMean().minus(obsMeanAdj);
//    
//    final Matrix W = F.times(Omega).times(F.transpose()).plus(Sigma);
//    final Matrix FG = F.times(G);
//    final Matrix A = FG.times(C).times(FG.transpose()).plus(W);
//    final Matrix Wtil =
//        A.transpose().solve(FG.times(C.transpose())).transpose();
//
//    final Vector mSmooth = m.plus(Wtil.times(y.minus(FG.times(m))));
//    final Matrix CSmooth =
//        C.minus(Wtil.times(A).times(Wtil.transpose()));
//    return new MultivariateGaussian(mSmooth, CSmooth);
//  }

  public static List<SimObservedValue<Vector, Matrix, Vector>> sampleDlm(Random random, int T, 
    MultivariateGaussian initialPrior, KalmanFilter filter) {
    return sampleDlm(random, T, initialPrior, filter, null);
  }

  /**
   * Sample observations and states from a DLM up to time {@code T}.
   * States are evolved according to the structural equations, then
   * error sampled from the model covariance is added.  Observations are from those
   * sampled states, transformed according to the DLM, with measurement error added.
   * 
   * @param random
   * @param T
   * @param function 
   * @return A list with <observation, state> samples
   */
  public static List<SimObservedValue<Vector, Matrix, Vector>> sampleDlm(Random random, int T, 
      MultivariateGaussian initialPrior, KalmanFilter filter, 
      Function<Matrix, Matrix> xGenerator) {
    List<SimObservedValue<Vector, Matrix, Vector>> results = Lists.newArrayList();

    Matrix curX = null;
    Vector currentState = initialPrior.getMean().clone();
    for (int i = 0; i < T; i++) {

      final Matrix G = filter.getModel().getA();
      Matrix modelCovSqrt = 
          ExtMatrixUtils.getCholR(filter.getModelCovariance());
      currentState = MultivariateGaussian.sample(G.times(currentState), modelCovSqrt, random);
      currentState.plusEquals(filter.getModel().getB().times(
          filter.getCurrentInput()));

      final Matrix F;
      if (xGenerator != null) {
        F = xGenerator.apply(curX);
        curX = F;
      } else {
        F = filter.getModel().getC();
        curX = F;
      }

      Vector observationMean = F.times(currentState);
      Matrix measurementCovSqrt = ExtMatrixUtils.getCholR(
          filter.getMeasurementCovariance());
      Vector observation = MultivariateGaussian.sample(observationMean, 
          measurementCovSqrt, random);
  
      results.add(SimObservedValue.<Vector, Matrix, Vector>create(
          i, observation, curX, currentState.clone()));
    }
    
    return results;
  }
}
