package com.statslibextensions.util;

public class ObservedValue<T, D> {
  
  final protected long time;
  final protected T observedValue;
  final protected D data;
  
  public static <T, D> ObservedValue<T, D> create(long time, T observedState, D data) {
    return new ObservedValue<T, D>(time, observedState, data);
  }

  /**
   * Creates an observed value with null data.
   * @param time
   * @param observedState
   * @return
   */
  public static <T> ObservedValue<T, Void> create(long time, T observedState) {
    return new ObservedValue<T, Void>(time, observedState, null);
  }

  /**
   * Creates an observed value with null data and the current time.
   * @param time
   * @param observedState
   * @return
   */
  public static <T> ObservedValue<T, Void> create(T observedState) {
    return new ObservedValue<T, Void>(System.currentTimeMillis(), observedState, null);
  }

  /**
   * Creates an observed value for the current time.
   * 
   * @param observedState
   * @param data
   * @return
   */
  public static <T, D> ObservedValue<T, D> create(T observedState, D data) {
    return new ObservedValue<T, D>(System.currentTimeMillis(), observedState, data);
  }

  protected ObservedValue(long time, T observedState, D data) {
    this.time = time;
    this.observedValue = observedState;
    this.data = data;
  }

  protected ObservedValue(long time, T observedState) {
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
