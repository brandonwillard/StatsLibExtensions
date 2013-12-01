package com.statslibextensions.math.matrix;

import static org.junit.Assert.*;

import java.util.Random;

import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;

import org.junit.Assert;
import org.junit.Test;

public class MonteCarloMatrixSolverTest {

  /**
   * Halton's original test problem.
   */
  @Test
  public void test1() {
    Random rng = new Random(23582958l);
    int m = 5;
    final Matrix H = MatrixFactory.getDefault().createMatrix(m, m);
    final Vector x = VectorFactory.getDefault().createVector(m);
    for (int i = 0; i < m; i++) {
      x.setElement(i, 1d/(2.25d - 1.45d * i/m));
      for (int j = 0; j < m; j++) {
        H.setElement(i, j, 0.9d/(m + i + j + 2d));
      }
    }
    final Vector a = x.minus(H.times(x));

    final Vector xEst = MonteCarloMatrixSolver.directSolve(H, a, 
        1e-1, rng);
    Assert.assertTrue(x.equals(xEst, 1e-2));
  }

  /**
   * A random system...
   */
  @Test
  public void test2() {
    Random rng = new Random(23582958l);
    int m = 5;
    final Matrix A = MatrixFactory.getDenseDefault().createUniformRandom(m, m, 0, 1, rng);
    final Vector x = VectorFactory.getDenseDefault().createUniformRandom(m, 0, 1, rng);
    final Vector b = A.times(x);
    final Vector xEst = MonteCarloMatrixSolver.directSolve(A, b, 
        MatrixFactory.getDiagonalDefault().createIdentity(m, m), 
        1e-2, rng);
    Assert.assertTrue(x.equals(xEst, 1e-3));
  }

}
