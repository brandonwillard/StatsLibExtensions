package com.statslibextensions.math;

import gov.sandia.cognition.math.MutableDouble;

public class MutableDoubleCount extends MutableDouble {

  private static final long serialVersionUID = -6936453778285494680L;

  public int count = 0;

  public MutableDoubleCount(double value) {
    super(value);
    this.count++;
  }

  public MutableDoubleCount(double value, int count) {
    super(value);
    this.count = count;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!super.equals(obj)) {
      return false;
    }
    if (this.getClass() != obj.getClass()) {
      return false;
    }
    final MutableDoubleCount other = (MutableDoubleCount) obj;
    if (this.count != other.count) {
      return false;
    }
    return true;
  }

  public int getCount() {
    return this.count;
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = super.hashCode();
    result = prime * result + this.count;
    return result;
  }

  public void plusEquals(double value) {
    this.value += value;
    this.count++;
  }

  public void plusEquals(double value, int count) {
    this.value += value;
    this.count += count;
  }

  public void set(double value) {
    this.set(value, 1);
  }

  public void set(double value, int count) {
    this.value = value;
    this.count = count;
  }

  @Override
  public String toString() {
    return "MutableDoubleCount [count=" + this.count + ", value="
        + this.value + "]";
  }
}