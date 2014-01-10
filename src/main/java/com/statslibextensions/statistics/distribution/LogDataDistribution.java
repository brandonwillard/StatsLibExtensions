package com.statslibextensions.statistics.distribution;

import gov.sandia.cognition.collection.ScalarMap;
import gov.sandia.cognition.factory.Factory;
import gov.sandia.cognition.learning.algorithm.AbstractBatchAndIncrementalLearner;
import gov.sandia.cognition.math.MathUtil;
import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.statistics.AbstractDataDistribution;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.DistributionEstimator;
import gov.sandia.cognition.statistics.DistributionWeightedEstimator;
import gov.sandia.cognition.statistics.ProbabilityMassFunctionUtil;
import gov.sandia.cognition.util.AbstractCloneableSerializable;
import gov.sandia.cognition.util.ArgumentChecker;
import gov.sandia.cognition.util.WeightedValue;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Doubles;
import com.statslibextensions.math.ExtLogMath;

/**
 * Copy of DefaultDataDistribution, but for log-scale weights.
 * 
 * @author bwillard
 * 
 * @param <KeyType>
 */
public class LogDataDistribution<KeyType> extends
    AbstractDataDistribution<KeyType> {

  /**
   * A factory for {@code LogDataDistribution} objects using some
   * given initial capacity for them.
   * 
   * @param <DataType>
   *          The type of data for the factory.
   */
  public static class DefaultFactory<DataType> extends
      AbstractCloneableSerializable implements
      Factory<LogDataDistribution<DataType>> {

    /**
       * 
       */
    private static final long serialVersionUID = 681699655965182747L;
    /** The initial domain capacity. */
    protected int initialDomainCapacity;

    /**
     * Creates a new {@code DefaultFactory} with a default initial domain
     * capacity.
     */
    public DefaultFactory() {
      this(LogDataDistribution.DEFAULT_INITIAL_CAPACITY);
    }

    /**
     * Creates a new {@code DefaultFactory} with a given initial domain
     * capacity.
     * 
     * @param initialDomainCapacity
     *          The initial capacity for the domain. Must be positive.
     */
    public DefaultFactory(final int initialDomainCapacity) {
      super();

      this.setInitialDomainCapacity(initialDomainCapacity);
    }

    @Override
    public LogDataDistribution<DataType> create() {
      // Create the histogram.
      return new LogDataDistribution<DataType>(
          this.getInitialDomainCapacity());
    }

    /**
     * Gets the initial domain capacity.
     * 
     * @return The initial domain capacity. Must be positive.
     */
    public int getInitialDomainCapacity() {
      return this.initialDomainCapacity;
    }

    /**
     * Sets the initial domain capacity.
     * 
     * @param initialDomainCapacity
     *          The initial domain capacity. Must be positive.
     */
    public void setInitialDomainCapacity(
      final int initialDomainCapacity) {
      ArgumentChecker.assertIsPositive("initialDomainCapacity",
          initialDomainCapacity);
      this.initialDomainCapacity = initialDomainCapacity;
    }

  }

  /**
   * Estimator for a LogDataDistribution
   * 
   * @param <KeyType>
   *          Type of Key in the distribution
   */
  public static class Estimator<KeyType>
      extends
      AbstractBatchAndIncrementalLearner<KeyType, LogDataDistribution.PMF<KeyType>>
      implements
      DistributionEstimator<KeyType, LogDataDistribution.PMF<KeyType>> {

    private static final long serialVersionUID = 8787720132790311008L;

    /**
     * Default constructor
     */
    public Estimator() {
      super();
    }

    @Override
    public Estimator<KeyType> clone() {
      final Estimator<KeyType> clone =
          (Estimator<KeyType>) super.clone();
      return clone;
    }

    @Override
    public LogDataDistribution.PMF<KeyType>
        createInitialLearnedObject() {
      return new LogDataDistribution.PMF<KeyType>();
    }

    @Override
    public void update(
      final LogDataDistribution.PMF<KeyType> target,
      final KeyType data) {
      target.increment(data, 0d);
    }

  }

  /**
   * PMF of the LogDataDistribution
   * 
   * @param <KeyType>
   *          Type of Key in the distribution
   */
  public static class PMF<KeyType> extends
      LogDataDistribution<KeyType> implements
      DataDistribution.PMF<KeyType> {

    /**
       * 
       */
    private static final long serialVersionUID = 355507596128913991L;

    /**
     * Default constructor
     */
    public PMF() {
      super();
    }

    /**
     * Copy constructor
     * 
     * @param other
     *          ScalarDataDistribution to copy
     */
    public PMF(final DataDistribution<KeyType> other) {
      super(other);
    }

    /**
     * Creates a new instance of LogDataDistribution
     * 
     * @param initialCapacity
     *          Initial capacity of the Map
     */
    public PMF(int initialCapacity) {
      super(initialCapacity);
    }

    /**
     * Creates a new instance of ScalarDataDistribution
     * 
     * @param data
     *          Data to create the distribution
     */
    public PMF(final Iterable<? extends KeyType> data) {
      super(data);
    }

    @Override
    public Double evaluate(KeyType input) {
      return this.getFraction(input);
    }

    @Override
    public LogDataDistribution.PMF<KeyType>
        getProbabilityFunction() {
      return this;
    }

    @Override
    public double logEvaluate(KeyType input) {
      return this.getLogFraction(input);
    }

  }

  /**
   * A weighted estimator for a LogDataDistribution
   * 
   * @param <KeyType>
   *          Type of Key in the distribution
   */
  public static class WeightedEstimator<KeyType>
      extends
      AbstractBatchAndIncrementalLearner<WeightedValue<? extends KeyType>, LogDataDistribution.PMF<KeyType>>
      implements
      DistributionWeightedEstimator<KeyType, LogDataDistribution.PMF<KeyType>> {

    /**
       * 
       */
    private static final long serialVersionUID =
        -9067384837227173014L;


    /**
     * Default constructor
     */
    public WeightedEstimator() {
      super();
    }

    @Override
    public LogDataDistribution.PMF<KeyType>
        createInitialLearnedObject() {
      return new LogDataDistribution.PMF<KeyType>();
    }

    @Override
    public void update(
      final LogDataDistribution.PMF<KeyType> target,
      final WeightedValue<? extends KeyType> data) {
      Preconditions.checkArgument(data.getWeight() <= 0d);
      target.increment(data.getValue(), data.getWeight());
    }

  }

  /**
   * Default initial capacity, {@value} .
   */
  public static final int DEFAULT_INITIAL_CAPACITY = 10;

  /**
   * 
   */
  private static final long serialVersionUID = 6596579376152342279L;

  /**
   * Total of the counts in the distribution
   */
  protected double total;

  public static <T> LogDataDistribution<T> create() {
    return new LogDataDistribution<T>();
  }

  public static <T> LogDataDistribution<T> create(int numEntries) {
    return new LogDataDistribution<T>(numEntries);
  }

  /**
   * Default constructor
   */
  public LogDataDistribution() {
    this(LogDataDistribution.DEFAULT_INITIAL_CAPACITY);
  }

  /**
   * Creates a new instance of LogDataDistribution
   * 
   * @param other
   *          DataDistribution to copy
   */
  public LogDataDistribution(
    final DataDistribution<? extends KeyType> other) {
    this(new LinkedHashMap<KeyType, MutableDouble>(other.size()),
        Double.NEGATIVE_INFINITY);
    this.incrementAll(other);
  }

  /**
   * Creates a new instance of LogDataDistribution
   * 
   * @param initialCapacity
   *          Initial capacity of the Map
   */
  public LogDataDistribution(int initialCapacity) {
    this(new LinkedHashMap<KeyType, MutableDouble>(initialCapacity),
        Double.NEGATIVE_INFINITY);
  }

  /**
   * Creates a new instance of ScalarDataDistribution
   * 
   * @param data
   *          Data to create the distribution
   */
  public LogDataDistribution(
    final Iterable<? extends KeyType> data) {
    this();
    this.incrementAll(data);
  }

  public LogDataDistribution(
    final Map<KeyType, MutableDouble> map, final double total) {
    super(map);
    this.total = total;
  }

  @Override
  public void clear() {
    super.clear();
    this.total = Double.NEGATIVE_INFINITY;
  }

  @Override
  public LogDataDistribution<KeyType> clone() {
    final LogDataDistribution<KeyType> clone =
        new LogDataDistribution<KeyType>(this.size());
    for (final java.util.Map.Entry<KeyType, MutableDouble> entry : this.map
        .entrySet()) {
      clone.set(entry.getKey(), entry.getValue().doubleValue());
    }

    clone.total = this.total;
    return clone;
  }

  public void copyAll(DataDistribution<KeyType> posteriorDist) {
    for (final java.util.Map.Entry<KeyType, ? extends Number> entry : posteriorDist
        .asMap().entrySet()) {
      this.set(entry.getKey(), entry.getValue().doubleValue());
    }
  }

  @Override
  public double decrement(KeyType key) {
    return this.decrement(key, 0d);
  }

  @Override
  public double decrement(KeyType key, double value) {
    throw new UnsupportedOperationException("not supported for log-scale");
//    return super.decrement(key, value);
  }

  @Override
  public void decrementAll(Iterable<? extends KeyType> keys) {
    throw new UnsupportedOperationException("not supported for log-scale");
//    super.decrementAll(keys);
  }

  @Override
  public void decrementAll(ScalarMap<? extends KeyType> other) {
    throw new UnsupportedOperationException("not supported for log-scale");
//    super.decrementAll(other);
  }

  /**
   * This value does not take duplicates into account.
   */
  @Override
  public int getDomainSize() {
    return super.getDomainSize();
  }

  @Override
  public double getEntropy() {
    final double identity =
        Double.NEGATIVE_INFINITY;
    double entropy = identity;
    final double total = this.getTotal();
    final double denom =
        (Doubles.compare(total, identity) != 0) ? total : 0d;
    for (final ScalarMap.Entry<KeyType> entry : this.entrySet()) {
      final double p = entry.getValue() - denom;
      if (Doubles.compare(p, identity) != 0) {
        entropy =
            ExtLogMath
                .subtract(entropy, p + p / MathUtil.log2(Math.E));
      }
    }
    return entropy;
  }

  @Override
  public
      DistributionEstimator<KeyType, ? extends DataDistribution<KeyType>>
      getEstimator() {
    return new LogDataDistribution.Estimator<KeyType>();
  }

  @Override
  public double getFraction(KeyType key) {
    return Math.exp(this.getLogFraction(key));
  }

  @Override
  public double getLogFraction(KeyType key) {
    Preconditions.checkArgument(this.containsKey(key));
    final double keyVal = this.get(key);
    return keyVal - this.getTotal();
  }

  /**
   * Gets the average value of all keys in the distribution, that is, the total
   * value divided by the number of keys (even zero-value keys)
   * 
   * @return Average value of all keys in the distribution
   */
  public double getMeanValue() {
    final int ds = this.getDomainSize();
    if (ds > 0) {
      return Math.exp(this.getTotal()) / ds;
    } else {
      return 0.0;
    }
  }

  @Override
  public DataDistribution.PMF<KeyType> getProbabilityFunction() {
    return new LogDataDistribution.PMF<KeyType>(this);
  }

  @Override
  public double getTotal() {
    return this.total;
  }

  @Override
  public double increment(KeyType key) {
    return this.increment(key, 0d);
  }

  /**
   * Increments by an existing key by value, or adds a new key with the given value.<br>
   * 
   * @param key
   * @param value
   * @param count
   * @return
   */
  @Override
  public double increment(KeyType key, final double value) {
    // TODO FIXME terrible hack!
    final MutableDouble entry = this.map.get(key);
    double newValue;
    final double identity = Double.NEGATIVE_INFINITY;
    if (entry == null) {
      if (value > identity) {
        this.map.put(key, new MutableDouble(value));
      } 
      newValue = value;
    } else {
      final double sum = ExtLogMath.add(entry.value, value);
      if (sum >= identity) {
        entry.setValue(sum);
      } else {
        entry.setValue(identity);
      }
      newValue = entry.value;
    }
    this.total = ExtLogMath.add(this.total, value);
    return newValue;
  }

  @Override
  public void incrementAll(Iterable<? extends KeyType> keys) {
    super.incrementAll(keys);
  }

  @Override
  public void incrementAll(ScalarMap<? extends KeyType> other) {
    super.incrementAll(other);
  }

  @Override
  public KeyType sample(Random random) {
    double w = random.nextDouble();
    for (final ScalarMap.Entry<KeyType> entry : this.entrySet()) {
      w -= this.getFraction(entry.getKey());
      if (w <= 0d) {
        return entry.getKey();
      }
    }
    return null;
  }

  @Override
  public ArrayList<KeyType> sample(Random random, int numSamples) {
    // Compute the cumulative weights
    final int size = this.getDomainSize();
    final double[] cumulativeWeights = new double[size];
    double cumulativeSum = 0d;
    final ArrayList<KeyType> domain = new ArrayList<KeyType>(size);
    int index = 0;
    for (final ScalarMap.Entry<KeyType> entry : this.entrySet()) {
      domain.add(entry.getKey());
      final double value = entry.getValue();
      cumulativeSum += Math.exp(value);
      cumulativeWeights[index] = cumulativeSum;
      index++;
    }

    return ProbabilityMassFunctionUtil.sampleMultiple(
        cumulativeWeights, cumulativeSum, domain, random,
        numSamples);
  }

  /**
   * Adds a new key with the given value, or
   * resets the value of an existing key.
   */
  @Override
  public void set(final KeyType key, final double totalValue) {
    // TODO FIXME terrible hack!
    final MutableDouble entry = this.map.get(key);
    final double identity = Double.NEGATIVE_INFINITY;
    if (entry == null) {
      // Only need to allocate if it's not null
      if (totalValue > identity) {
        this.map.put(key, new MutableDouble(totalValue));
        this.total = ExtLogMath.add(this.total, totalValue);
      }
    } else if (totalValue > identity) {
      if (entry.value > totalValue) {
        final double totalsSum =
            ExtLogMath.add(this.total, totalValue);
        Preconditions.checkState(totalsSum >= entry.value);
        this.total = ExtLogMath.subtract(totalsSum, entry.value);
      } else {
        this.total =
            ExtLogMath.add(this.total,
                ExtLogMath.subtract(totalValue, entry.value));
      }
      entry.setValue(totalValue);
    } else {
      Preconditions.checkArgument(Doubles.compare(totalValue,
          identity) == 0);
      entry.setValue(identity);
    }
  }

  public double adjust(KeyType key, final double value) {
    final MutableDouble entry = this.map.get(key);
    double newValue;
    double delta;
    final double identity = Double.NEGATIVE_INFINITY;
    if (entry == null) {
      if (value > identity) {
        this.map.put(key, new MutableDouble(value));
        delta = value;
      } else {
        delta = identity;
      }
      newValue = value;
    } else {
      final double sum = ExtLogMath.add(entry.value, value);

      // TODO XXX FIXME this needs to be checked/fixed
      if (sum >= identity) {
        delta = value;
        entry.setValue(sum);
      } else if (sum == identity) {
        delta = identity;
        entry.setValue(sum);
        this.map.remove(key);
      } else {
        delta = -entry.value;
        entry.setValue(identity);
      }
      newValue = entry.value;
    }

    if (delta == identity) 
      this.total = ExtLogMath.subtract(this.total, value);
    else
      this.total = ExtLogMath.add(this.total, value);
    
    return newValue;
  }
  

}
