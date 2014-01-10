package com.statslibextensions.util;

import gov.sandia.cognition.util.DefaultWeightedValue;
import gov.sandia.cognition.util.Weighted;

public class ExtDefaultWeightedValue<ValueType> extends DefaultWeightedValue<ValueType>
implements Comparable<Weighted>{

  private static final long serialVersionUID = 4373636363266356173L;

  public ExtDefaultWeightedValue(ValueType value, double weight) {
    super(value, weight);
  }

  @Override
  public int compareTo(Weighted o) {
//    return DefaultWeightedValue.WeightComparator.getInstance().compare(this, o);
    return Double.compare(this.weight, o.getWeight());
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("ExtDefaultWeightedValue [value=").append(this.value)
        .append("\n\t weight=").append(this.weight).append("]");
    return builder.toString();
  }

  public static <ValueType> ExtDefaultWeightedValue<ValueType> create(
      final ValueType value, final double weight) {
    return new ExtDefaultWeightedValue<ValueType>(value, weight);
  }
}
