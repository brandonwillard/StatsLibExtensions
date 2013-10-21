package com.statslibextensions.util;

import com.google.common.base.Preconditions;

import no.uib.cipr.matrix.DenseCholesky;
import no.uib.cipr.matrix.DenseMatrix;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.mtj.DenseMatrixFactoryMTJ;
import gov.sandia.cognition.math.matrix.mtj.DiagonalMatrixFactoryMTJ;
import gov.sandia.cognition.math.matrix.mtj.DiagonalMatrixMTJ;
import gov.sandia.cognition.math.matrix.mtj.decomposition.SingularValueDecompositionMTJ;

public class ExtMatrixUtils {

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
   * Returns, for A, an R s.t. R<sup>T</sup> * R = A
   * 
   * @param matrix
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

  public static Matrix getDiagonalInverse(Matrix S, double lowerTolerance) {
    Preconditions.checkArgument(lowerTolerance >= 0d);
    final Matrix result =
        MatrixFactory.getDefault().createMatrix(S.getNumColumns(), S.getNumColumns());
    for (int i = 0; i < Math.min(S.getNumColumns(), S.getNumRows()); i++) {
      final double sVal = S.getElement(i, i);
      if (Math.abs(sVal) > lowerTolerance) result.setElement(i, i, 1d / sVal);
    }
    return result;
  }

  public static Matrix getDiagonalSqrt(Matrix mat, double tolerance) {
    Preconditions.checkArgument(tolerance > 0d);
    final Matrix result = mat.clone();
    for (int i = 0; i < Math.min(result.getNumColumns(), result.getNumRows()); i++) {
      final double sqrt = Math.sqrt(result.getElement(i, i));
      if (sqrt > tolerance) result.setElement(i, i, sqrt);
    }
    return result;
  }

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

  public static Matrix rootOfSemiDefinite(Matrix matrix) {
    return ExtMatrixUtils.rootOfSemiDefinite(matrix, false, -1);
  }

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

}
