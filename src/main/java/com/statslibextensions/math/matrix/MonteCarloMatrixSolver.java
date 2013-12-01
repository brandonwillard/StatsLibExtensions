package com.statslibextensions.math.matrix;

import java.util.Random;

import com.google.common.base.Preconditions;

import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;


/**
 * 
 * Halton's sequential MC solution for
 * A x = b, where A is m x m positive definite
 * Let a = G*b, H = I - G*A and p<sub>j</sub> = 1/m, P<sub>ij</sub> = (1-w)/m
 * Sample multiple sequences {g<sub>1</sub>, g<sub>2</sub>, ..., g<sub>r</sub>}
 * from the markov chain produced by p<sub>j</sub> and P<sub>ij</sub>.  
 * Sampling proceeds until the "stopping index"
 * \sigma is reached, i.e. when g<sub>t</sub> = 0, so that
 * t = \sigma + 1.<br>
 * Using one of the two estimators
 * <ul>
 * <li> Direct: 
 *   H<sub>i,g<sub>1</sub></sub>* H<sub>g<sub>1</sub>,g<sub>2</sub></sub>...H<sub>g<sub>r-1</sub>, g<sub>r</sub></sub> a<sub>g<sub>r</sub></sub>/
 *      (p<sub>g<sub>1</sub></sub> P<sub>g<sub>1</sub>, g<sub>2</sub></sub> ... P<sub>g<sub>r-1</sub>, g<sub>r</sub></sub>)
 * </li>
 * <li> Adjoint: 
 *   H<sub>i,g<sub>r</sub></sub>* H<sub>g<sub>r</sub>,g<sub>r-1</sub></sub>...H<sub>g<sub>2</sub>, g<sub>1</sub></sub> a<sub>g<sub>1</sub></sub>/
 *      (p<sub>g<sub>1</sub></sub> P<sub>g<sub>1</sub>, g<sub>2</sub></sub> ... P<sub>g<sub>r-1</sub>, g<sub>r</sub></sub>)
 * </li>
 * </ul>
 * and averaging over the sampled sequences, we
 * obtain an estimate of x.
 * 
 * @author bwillar0
 *
 */
public class MonteCarloMatrixSolver {

  /**
   * Direct sequential MC solver for <code>A*x = b</code>.
   *  
   * @param A
   * @param b
   * @param G
   * @param stop
   * @param rng
   * @return
   */
  public static Vector directSolve(Matrix A, Vector b, Matrix G, double stop, Random rng) {
    final Matrix H = MatrixFactory.getDefault().createIdentity(
        G.getNumRows(), A.getNumColumns());
    H.minusEquals(G.times(A));
    final Vector a = G.times(b);
    return directSolve(H, a, stop, rng);
  }

  /**
   * Direct sequential MC solver for <code>A*x = b</code>.
   * 
   * @param H = I - G*A 
   * @param a = G*b
   * @param stop
   * @param rng
   * @return
   */
  public static Vector directSolve(Matrix H, Vector a, double stop, Random rng) {

    Preconditions.checkArgument(stop < 1d && stop >= 0d);
    int m = H.getNumRows();
    final double mult = m/(1d - stop);
  
    double xi = rng.nextDouble();
    int j = (int)Math.ceil(m*xi) - 1;
    Vector fac = H.getColumn(j).scale(m);

    Vector g = a.clone();
    g.scaledPlusEquals(a.getElement(j), fac);

    double prod_dir = 1d;

    while (true) {
      int h = j;
      final double xii = rng.nextDouble();
      if (xii < stop) 
        break;
      j = (int)Math.ceil(mult * (xii - stop)) - 1;
      prod_dir *= mult * H.getElement(h,j);
      g.scaledPlusEquals(a.getElement(j) * prod_dir, fac);
    }
  
    return g;
  }

}
