package com.statslibextensions.statistics.distribution;

import gov.sandia.cognition.statistics.DataDistribution;

import java.util.Map;

import com.statslibextensions.statistics.distribution.CountedDataDistribution;

/**
 * Just a wrapper that carries water-filling debug information.
 * 
 * @author bwillard
 *
 * @param <T>
 */
public class WFCountedDataDistribution<T> extends
    CountedDataDistribution<T> {
  
  boolean wasWaterFillingApplied = false;
  
  public static <T> WFCountedDataDistribution<T> create(boolean isLogScale) {
    return new WFCountedDataDistribution<T>(isLogScale);
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("WFCountedDataDistribution [wasWaterFillingApplied=")
        .append(this.wasWaterFillingApplied).append(", isLogScale=")
        .append(this.isLogScale).append(", map=").append(this.map).append("]");
    return builder.toString();
  }

  public static <T> WFCountedDataDistribution<T> create(int initialCapacity, boolean isLogScale) {
    return new WFCountedDataDistribution<T>(initialCapacity, isLogScale);
  }

  public WFCountedDataDistribution(boolean isLogScale) {
    super(isLogScale);
    // TODO Auto-generated constructor stub
  }

  public WFCountedDataDistribution(DataDistribution other,
    boolean isLogScale) {
    super(other, isLogScale);
    // TODO Auto-generated constructor stub
  }

  public WFCountedDataDistribution(int initialCapacity,
    boolean isLogScale) {
    super(initialCapacity, isLogScale);
    // TODO Auto-generated constructor stub
  }

  public WFCountedDataDistribution(Iterable data, boolean isLogScale) {
    super(data, isLogScale);
    // TODO Auto-generated constructor stub
  }

  public WFCountedDataDistribution(Map map, double total,
    boolean isLogScale) {
    super(map, total, isLogScale);
    // TODO Auto-generated constructor stub
  }

  public boolean wasWaterFillingApplied() {
    return wasWaterFillingApplied;
  }

  public void setWasWaterFillingApplied(boolean wasWaterFillingApplied) {
    this.wasWaterFillingApplied = wasWaterFillingApplied;
  }

}
