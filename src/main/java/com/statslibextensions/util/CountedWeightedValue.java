package com.statslibextensions.util;

import gov.sandia.cognition.util.DefaultWeightedValue;
import gov.sandia.cognition.util.WeightedValue;

public class CountedWeightedValue<T> extends DefaultWeightedValue<T> {

  private static final long serialVersionUID = -2223108371382713360L;

  private int count = 0;

  public static <ValueType> CountedWeightedValue<ValueType> create(
      final ValueType value, final double weight) {
    return new CountedWeightedValue<ValueType>(value, weight);
  }

  public static <ValueType> CountedWeightedValue<ValueType> create(
      final ValueType value, final double weight, int count) {
    return new CountedWeightedValue<ValueType>(value, weight, count);
  }

  public CountedWeightedValue() {
    super();
  }

  public CountedWeightedValue(T value) {
    super(value);
  }

  public CountedWeightedValue(T value, double weight) {
    super(value, weight);
    this.count++;
  }

  public CountedWeightedValue(T value, double weight, int count) {
    super(value, weight);
    this.count = count;
  }

  public CountedWeightedValue(WeightedValue<? extends T> other) {
    super(other);
  }

  public int getCount() {
    return this.count;
  }

  @Override
  public String toString() {
    return "WrappedWeightedValue [count=" + this.count + ", value="
        + this.value + ", weight=" + this.weight + "]";
  }
}
