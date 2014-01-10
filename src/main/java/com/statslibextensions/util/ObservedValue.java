package com.statslibextensions.util;

public class ObservedValue<O, D> {
  
  public static class SimObservedValue<O, D, T> extends ObservedValue<O, D> {

    final protected T trueState;

    protected SimObservedValue(long time, O observedState, T trueState) {
      super(time, observedState);
      this.trueState = trueState;
    }

    protected SimObservedValue(long time, O observedState, D data, T trueState) {
      super(time, observedState, data);
      this.trueState = trueState;
    }

    public static <O, D, T> SimObservedValue<O, D, T> create(long time, O observedState, D data, T trueState) {
      return new SimObservedValue<O, D, T>(time, observedState, data, trueState);
    }
  
//    public static <O, T> SimObservedValue<O, Void, T> create(long time, O observedState, T trueValue) {
//      return new SimObservedValue<O, Void, T>(time, observedState, null, trueValue);
//    }
//  
//    public static <O, T> SimObservedValue<O, Void, T> create(O observedState, T trueState) {
//      return new SimObservedValue<O, Void, T>(System.currentTimeMillis(), observedState, null, trueState);
//    }
  
    public static <O, D, T> SimObservedValue<O, D, T> create(O observedState, D data, T trueState) {
      return new SimObservedValue<O, D, T>(System.currentTimeMillis(), observedState, data, trueState);
    }

    public T getTrueState() {
      return this.trueState;
    }

    @Override
    public String toString() {
      StringBuilder builder = new StringBuilder();
      builder.append("SimObservedValue [trueState=").append(this.trueState)
          .append(", time=").append(this.time).append(", observedValue=")
          .append(this.observedValue).append(", data=").append(this.data)
          .append("]");
      return builder.toString();
    }
    
  }
  
  final protected long time;
  final protected O observedValue;
  final protected D data;
  
  public static <O, D> ObservedValue<O, D> create(long time, O observedState, D data) {
    return new ObservedValue<O, D>(time, observedState, data);
  }

  /**
   * Creates an observed value with null data.
   * @param time
   * @param observedState
   * @return
   */
  public static <O> ObservedValue<O, Void> create(long time, O observedState) {
    return new ObservedValue<O, Void>(time, observedState, null);
  }

  /**
   * Creates an observed value with null data and the current time.
   * @param time
   * @param observedState
   * @return
   */
  public static <O> ObservedValue<O, Void> create(O observedState) {
    return new ObservedValue<O, Void>(System.currentTimeMillis(), observedState, null);
  }

  /**
   * Creates an observed value for the current time.
   * 
   * @param observedState
   * @param data
   * @return
   */
  public static <O, D> ObservedValue<O, D> create(O observedState, D data) {
    return new ObservedValue<O, D>(System.currentTimeMillis(), observedState, data);
  }

  protected ObservedValue(long time, O observedState, D data) {
    this.time = time;
    this.observedValue = observedState;
    this.data = data;
  }

  protected ObservedValue(long time, O observedState) {
    this.time = time;
    this.observedValue = observedState;
    this.data = null;
  }

  public long getTime() {
    return time;
  }

  public O getObservedValue() {
    return observedValue;
  }
  
  public D getObservedData() {
    return data;
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("ObservedValue [time=").append(this.time)
        .append(", observedValue=").append(this.observedValue)
        .append(", data=").append(this.data).append("]");
    return builder.toString();
  }
  
  
}
