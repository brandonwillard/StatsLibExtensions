package com.statslibextensions.util;

import com.google.common.base.Preconditions;
import com.statslibextensions.math.matrix.SvdMatrix;
import com.statslibextensions.math.matrix.decomposition.SimpleSingularValueDecomposition;

import no.uib.cipr.matrix.DenseCholesky;
import no.uib.cipr.matrix.DenseMatrix;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.matrix.decomposition.AbstractSingularValueDecomposition;
import gov.sandia.cognition.math.matrix.mtj.DenseMatrixFactoryMTJ;
import gov.sandia.cognition.math.matrix.mtj.DiagonalMatrixFactoryMTJ;
import gov.sandia.cognition.math.matrix.mtj.DiagonalMatrixMTJ;
import gov.sandia.cognition.math.matrix.mtj.decomposition.SingularValueDecompositionMTJ;

public class ExtMatrixUtils {

  public static Vector getDiagonal(Matrix matrix) {
    Vector result = VectorFactory.getDefault().createVector(matrix.getNumColumns());
    for (int i = 0; i < matrix.getNumColumns(); i++) {
      result.setElement(i, matrix.getElement(i, i));
    }
    return result;
  }

  /**
   * Computes the Penrose pseudo-inverse via SVD, reducing the resulting matrix's
   * to the effective rank (current effective zero is 1e-9).
   * 
   * @param matrix
   * @return
   */
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
   * Check for pos. definiteness using the singular values
   * (and an SVD if necessary).
   * 
   * @param covar
   * @return
   */
  public static boolean isPosSemiDefinite(Matrix covar) {
    if (covar instanceof SvdMatrix) {
      final AbstractSingularValueDecomposition svd =
          ((SvdMatrix) covar).getSvd();
      if (svd.getU().equals(svd.getVtranspose().transpose())) {
        return true;
      } else {
        return false;
      }
    } else {
      try {
        DenseCholesky.factorize(DenseMatrixFactoryMTJ.INSTANCE
            .copyMatrix(covar).getInternalMatrix());
      } catch (final IllegalArgumentException ex) {
        return false;
      }
      return true;
    }
  }

  /**
   * Returns R s.t. R<sup>T</sup>*R = A, for A > 0, using Cholesky decomp.
   * 
   * @param matrix A
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

  /**
   * Inverts the entries in the diagonal of the given matrix.
   * 
   * @param S
   * @param lowerTolerance
   * @return
   */
  public static Matrix getDiagonalInverse(Matrix S, double lowerTolerance) {
    Preconditions.checkArgument(lowerTolerance >= 0d);
    final Matrix result =
        MatrixFactory.getDiagonalDefault().createMatrix(S.getNumColumns(), S.getNumColumns());
    for (int i = 0; i < Math.min(S.getNumColumns(), S.getNumRows()); i++) {
      final double sVal = S.getElement(i, i);
      if (Math.abs(sVal) > lowerTolerance) result.setElement(i, i, 1d / sVal);
    }
    return result;
  }

  /**
   * Returns the sqrt. of the diagonal elements of the given matrix.
   * 
   * @param mat
   * @param tolerance
   * @return
   */
  public static Matrix getDiagonalSqrt(Matrix mat, double tolerance) {
    Preconditions.checkArgument(tolerance >= 0d);
    final Matrix result = 
        MatrixFactory.getDiagonalDefault().createMatrix(mat.getNumColumns(), mat.getNumColumns());
    for (int i = 0; i < Math.min(result.getNumColumns(), result.getNumRows()); i++) {
      final double sqrt = Math.sqrt(result.getElement(i, i));
      if (sqrt > tolerance) result.setElement(i, i, sqrt);
    }
    return result;
  }

  /**
   * Returns the square of the diagonal elements of the given matrix.
   * 
   * @param S
   * @param tolerance
   * @return
   */
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

  /**
   * @see ExtMatrixUtils#rootOfSemiDefinite(Matrix, boolean, int)
   * 
   * @param matrix
   * @return
   */
  public static Matrix rootOfSemiDefinite(Matrix matrix) {
    return ExtMatrixUtils.rootOfSemiDefinite(matrix, false, -1);
  }

  /**
   * 
   * Returns R s.t. R<sup>T</sup>*R = A with R reduced in dimension by
   * the given value of rank or A's effective rank, when the given rank <= 0.
   * 
   * @param matrix A
   * @param effRankDimResult
   * @param rank
   * @return
   */
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

  /**
   * Computes G*C*G^T + W in a numerically stable way for symmetric matrices C
   * and W.
   * 
   * @param C
   * @param W
   * @param G
   * @return
   */
  public static SvdMatrix symmetricSvdAdd(SvdMatrix C, SvdMatrix W,
    Matrix G) {
    final AbstractSingularValueDecomposition svdC = C.getSvd();
    final Matrix SUG =
        ExtMatrixUtils.getDiagonalSqrt(svdC.getS(), 1e-7)
            .times(svdC.getU().transpose()).times(G.transpose());
    final Matrix Nw =
        ExtMatrixUtils.getDiagonalSqrt(W.getSvd().getS(), 1e-7)
            .times(W.getSvd().getU().transpose());
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
    return new SvdMatrix(svdR);
  }
}
