package com.statslibextensions.statistics.distribution;

import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.AbstractDataDistribution;
import gov.sandia.cognition.statistics.ClosedFormComputableDiscreteDistribution;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.DistributionEstimator;
import gov.sandia.cognition.statistics.distribution.DefaultDataDistribution;
import gov.sandia.cognition.util.ObjectUtil;

import java.util.Collections;

public class DeterministicDataDistribution<T> extends
    AbstractDataDistribution<T> implements
    ClosedFormComputableDiscreteDistribution<T> {

  final protected static MutableDouble internalValue =
      new MutableDouble(1d);

  private static final long serialVersionUID = 5553981567680543038L;

  protected T element;

  public DeterministicDataDistribution(T element) {
    super(Collections.singletonMap(element,
        DeterministicDataDistribution.internalValue));
    this.element = element;
  }

  @Override
  public DeterministicDataDistribution<T> clone() {
    final DeterministicDataDistribution<T> clone =
        new DeterministicDataDistribution<T>(
            ObjectUtil.cloneSmart(this.element));
    return clone;
  }

  /**
   * Warning: no-op
   */
  @Override
  public void convertFromVector(Vector parameters) {

  }

  /**
   * Warning: no-op
   */
  @Override
  public Vector convertToVector() {
    return null;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (!(obj instanceof DeterministicDataDistribution)) {
      return false;
    }
    final DeterministicDataDistribution other =
        (DeterministicDataDistribution) obj;
    if (this.element == null) {
      if (other.element != null) {
        return false;
      }
    } else if (!this.element.equals(other.element)) {
      return false;
    }
    return true;
  }

  public T getElement() {
    return this.element;
  }

  @Override
  public DistributionEstimator<T, ? extends DataDistribution<T>>
      getEstimator() {
    return null;
  }

  @Override
  public T getMean() {
    return this.element;
  }

  @Override
  public PMF<T> getProbabilityFunction() {
    return new DefaultDataDistribution.PMF<T>(this);
  }

  @Override
  public double getTotal() {
    return DeterministicDataDistribution.internalValue.doubleValue();
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result =
        prime * result
            + ((this.element == null) ? 0 : this.element.hashCode());
    return result;
  }

  public void setElement(T priorPredRunState) {
    this.map =
        Collections.singletonMap(priorPredRunState,
            DeterministicDataDistribution.internalValue);
    this.element = priorPredRunState;
  }

}
