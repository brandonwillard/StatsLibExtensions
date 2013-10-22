package com.statslibextensions.util;

public class ObservedValue<T, D> {
  
  final protected long time;
  final protected T observedValue;
  final protected D data;

  public ObservedValue(long time, T observedState, D data) {
    this.time = time;
    this.observedValue = observedState;
    this.data = data;
  }

  public ObservedValue(long time, T observedState) {
    this.time = time;
    this.observedValue = observedState;
    this.data = null;
  }

  public long getTime() {
    return time;
  }

  public T getObservedValue() {
    return observedValue;
  }
  
  public D getObservedData() {
    return data;
  }

  @Override
  public String toString() {
    return "ObservedValue [time=" + time + ", observedValue=" + observedValue
        + "]";
  }
  
  
}
