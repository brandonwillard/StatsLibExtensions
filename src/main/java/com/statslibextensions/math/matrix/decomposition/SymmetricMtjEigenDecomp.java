package com.statslibextensions.math.matrix.decomposition;

import com.google.common.base.Preconditions;
import com.statslibextensions.statistics.ExtSamplingUtils;

import gov.sandia.cognition.math.ComplexNumber;
import gov.sandia.cognition.math.OperationNotConvergedException;
import gov.sandia.cognition.math.matrix.DiagonalMatrix;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.decomposition.AbstractEigenDecomposition;
import gov.sandia.cognition.math.matrix.mtj.DenseMatrix;
import gov.sandia.cognition.math.matrix.mtj.DenseMatrixFactoryMTJ;
import gov.sandia.cognition.math.matrix.mtj.DenseVectorFactoryMTJ;
import gov.sandia.cognition.math.matrix.mtj.DiagonalMatrixFactoryMTJ;
import gov.sandia.cognition.math.matrix.mtj.decomposition.EigenDecompositionRightMTJ;
import gov.sandia.cognition.util.ArrayIndexSorter;
import no.uib.cipr.matrix.EVD;
import no.uib.cipr.matrix.LowerSymmDenseMatrix;
import no.uib.cipr.matrix.Matrices;
import no.uib.cipr.matrix.SymmDenseEVD;

public class SymmetricMtjEigenDecomp {

  private DiagonalMatrix eigenValues;
  private DenseMatrix eigenVectors;
  private int nonZeroEigenValues = 0;
  private double[] eigenValuesArray;
  
  /**
   * Creates a new instance of EigenDecompositionRightMTJ.
   * 
   * @param matrix
   *          DenseMatrix to compute the right EVD of
   * @throws OperationNotConvergedException
   *           If the operation does not converge.
   */
  private SymmetricMtjEigenDecomp(final DenseMatrix matrix, double eigenValTol)
      throws OperationNotConvergedException {
    SymmDenseEVD mtjEVD = new SymmDenseEVD(matrix.getNumRows(), false);
    try {
      mtjEVD.factor(new no.uib.cipr.matrix.LowerSymmDenseMatrix(matrix
          .getInternalMatrix()));
    } catch (no.uib.cipr.matrix.NotConvergedException e) {
      throw new OperationNotConvergedException(e.getMessage());
    }

    final int N = mtjEVD.getEigenvalues().length;
    this.eigenValuesArray = mtjEVD.getEigenvalues();
//    no.uib.cipr.matrix.DenseMatrix evs = mtjEVD.getEigenvectors();
    this.eigenValues = DiagonalMatrixFactoryMTJ.INSTANCE.createMatrix(N);
    this.eigenVectors = DenseMatrixFactoryMTJ.INSTANCE.createMatrix(N, N);

    int[] indices = ArrayIndexSorter.sortArrayDescending(mtjEVD.getEigenvalues());
    for (int j = 0 ; j < indices.length; j++) {
      final int j2 = indices[j]; 
      final double thisEigenVal = mtjEVD.getEigenvalues()[j2];
      if (Math.abs(thisEigenVal) > eigenValTol) {
        this.nonZeroEigenValues++;
        this.eigenValues.setElement(j, thisEigenVal);
      }
      for (int i = 0; i < N; i++) {
        this.eigenVectors.setElement(i, j, mtjEVD.getEigenvectors().get(i, j2));;
      }
    }
  }

  private void factor(LowerSymmDenseMatrix lowerSymmDenseMatrix) {
    
  }

  public static SymmetricMtjEigenDecomp create(final DenseMatrix matrix, double eigenValTol)
      throws OperationNotConvergedException {
    Preconditions.checkArgument(eigenValTol > 0d);
    return new SymmetricMtjEigenDecomp(matrix, eigenValTol);
  }

  public int getNonZeroEigenValues() {
    return this.nonZeroEigenValues;
  }

  public Matrix getEigenVectors() {
    return this.eigenVectors;
  }

  public Matrix getEigenValues() {
    return this.eigenValues;
  }

  public double[] getEigenValuesArray() {
    return this.eigenValuesArray;
  }


}
