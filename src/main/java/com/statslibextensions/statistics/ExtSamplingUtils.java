package com.statslibextensions.statistics;

import gov.sandia.cognition.collection.ArrayUtil;
import gov.sandia.cognition.math.LogMath;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.util.DefaultWeightedValue;
import gov.sandia.cognition.util.Weighted;
import gov.sandia.cognition.util.WeightedValue;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.TreeMap;
import java.util.TreeSet;

import org.apache.log4j.Logger;
import com.google.common.base.Preconditions;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.MinMaxPriorityQueue;
import com.google.common.collect.Multiset;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;
import com.google.common.primitives.Doubles;
import com.statslibextensions.math.ExtLogMath;
import com.statslibextensions.statistics.distribution.CountedDataDistribution;
import com.statslibextensions.statistics.distribution.WFCountedDataDistribution;
import com.statslibextensions.util.ComparableWeighted;
import com.statslibextensions.util.ExtDefaultWeightedValue;

public class ExtSamplingUtils {
  
  final public static Logger log = Logger
      .getLogger(ExtSamplingUtils.class);

  /**
   * "Low variance"/systematic categorical sampler. 
   * Follows Thrun's example in Probabilistic Robots.
   * 
   * @param rng
   * @param particles
   * @param M
   * @return
   */
  public static <SupportType extends Weighted> Multiset<SupportType> lowVarianceSampler(Random rng,
      Multiset<SupportType> particles, double M) {
    Preconditions.checkArgument(particles.size() > 0);

    final Multiset<SupportType> resampled = HashMultiset.create((int) M);
    final double logM = Math.log(M);
    final double r = Math.log(rng.nextDouble());
    final Iterator<SupportType> pIter = particles.iterator();
    SupportType p = pIter.next();
    double c = p.getWeight() - Math.log(particles.count(p));
    for (int m = 0; m < M; ++m) {
      final double U = LogMath.add(r, Math.log(m)) - logM;
      while (U > c && pIter.hasNext()) {
        p = pIter.next();
        c = LogMath.add(p.getWeight() - Math.log(particles.count(p)), c);
      }
      resampled.add(p);
    }
    return resampled;
  }
  
  public static <T> TreeSet<WeightedValue<T>> getLogWeighedList(DataDistribution<T> dist) {
    TreeSet<WeightedValue<T>> result = Sets.newTreeSet(DefaultWeightedValue.WeightComparator.getInstance());
    for (T entry : dist.getDomain()) {
      result.add(ExtDefaultWeightedValue.create(entry, 
          dist.getLogFraction(entry)));
    }
    return result;
  }

  /**
   * See {@link #lowVarianceSampler}.
   * 
   * @param rng
   * @param particles
   * @param M
   * @return
   */
  public static <D> Multiset<D> lowVarianceSampler(
      DataDistribution<D> dist, Random rng, int M) {

    final Multiset<D> resampled = HashMultiset.create(M);
    final double logM = Math.log(M);
    final double r = Math.log(rng.nextDouble());
    final Iterator<? extends D> pIter = dist.getDomain().iterator();
    D p = pIter.next();
    double c = dist.getLogFraction(p);
    for (int m = 0; m < M; ++m) {
      final double U = LogMath.add(r,  Math.log(m)) - logM;
      while (U > c && pIter.hasNext()) {
        p = pIter.next();
        c = LogMath.add(dist.getLogFraction(p), c);
      }
      resampled.add(p);
    }
    return resampled;
  }

  /**
   * See {@link #lowVarianceSampler}.
   * 
   * @param rng
   * @param particles
   * @param M
   * @return
   */
  public static <D> Multiset<D> lowVarianceSampler(
      Collection<WeightedValue<D>> particles, Random rng, int M) {
    Preconditions.checkArgument(particles.size() > 0);

    final Multiset<D> resampled = HashMultiset.create(M);
    final double logM = Math.log(M);
    final double r = Math.log(rng.nextDouble());
    final Iterator<WeightedValue<D>> pIter = particles.iterator();
    WeightedValue<D> p = pIter.next();
    double c = p.getWeight();
    for (int m = 0; m < M; ++m) {
      final double U = LogMath.add(r,  Math.log(m)) - logM;
      while (U > c && pIter.hasNext()) {
        p = pIter.next();
        c = LogMath.add(p.getWeight(), c);
      }
      resampled.add(p.getValue());
    }
    return resampled;
  }

  /**
   * Water-filling resample.
   * 
   * @param logWeights
   * @param logWeightSum
   * @param domain
   * @param random
   * @param N number of samples to return
   * @return
   */
  public static <D> WFCountedDataDistribution<D> waterFillingResample(final double[] logWeights, 
    final double logWeightSum, final Collection<D> domain, final Random random, final int N) {
    Preconditions.checkArgument(domain.size() == logWeights.length,
        "number of log weights must be equal to the support size");
    Preconditions.checkArgument(logWeights.length >= N,
        "number of log weights must be >= N");

    final List<Double> nonZeroLogWeights = Lists.newArrayList();
    final List<Double> cumNonZeroLogWeights = Lists.newArrayList();
    final List<D> nonZeroObjects = Lists.newArrayList();
    final DataDistribution<D> nonZeroDist = CountedDataDistribution.create(true);
    double nonZeroTotal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < logWeights.length; i++) {
      final double normedLogWeight = logWeights[i] - logWeightSum;
      if (Double.compare(normedLogWeight, Double.NEGATIVE_INFINITY) > 0d) {
        D obj = Iterables.get(domain, i);
        nonZeroObjects.add(obj);
        nonZeroDist.increment(obj, normedLogWeight);
        nonZeroLogWeights.add(normedLogWeight);
        nonZeroTotal = ExtLogMath.add(nonZeroTotal, normedLogWeight);
        cumNonZeroLogWeights.add(nonZeroTotal);
      }
    }
    
    Preconditions.checkState(Math.abs(Iterables.getLast(cumNonZeroLogWeights)) < 1e-7,
        "log weights must be normalized by the give log weight sum");
    
    boolean wasWaterFillingApplied = false;
    List<Double> resultLogWeights;
    List<D> resultObjects;
    final int nonZeroCount = nonZeroLogWeights.size();
    Preconditions.checkState(nonZeroCount >= N);
    if (nonZeroCount == N) {
      /*
       * Do nothing but remove the zero weights
       */
      resultLogWeights = nonZeroLogWeights;
      resultObjects = nonZeroObjects;
      log.debug("removed zero weights");

    } else {

      final double logAlpha = findLogAlpha(Doubles.toArray(nonZeroLogWeights),
          nonZeroTotal, N);
      if (logAlpha == 0 || Double.isNaN(logAlpha)) {
        /*
         * Plain 'ol resample here, too.
         */
//        resultObjects = sampleNoReplaceMultipleLogScaleES(Doubles.toArray(nonZeroLogWeights), 
//            nonZeroObjects, random, N);
//        resultObjects = sampleReplaceCumulativeLogScale(
//            Doubles.toArray(cumNonZeroLogWeights), nonZeroObjects, random, N);
        resultObjects = Lists.newArrayList(lowVarianceSampler(nonZeroDist, random, N));
        resultLogWeights = Collections.nCopies(N, -Math.log(N));
        log.warn("logAlpha = 0");
      } else {

        List<Double> logPValues = Lists.newArrayListWithCapacity(nonZeroCount);
        List<Double> keeperLogWeights = Lists.newArrayList();
        List<D> keeperObjects = Lists.newArrayList();
        List<Double> belowLogWeights = Lists.newArrayList();
        List<D> belowObjects = Lists.newArrayList();
        double belowPTotal = Double.NEGATIVE_INFINITY;
        for (int j = 0; j < nonZeroLogWeights.size(); j++) {
          final double logQ = nonZeroLogWeights.get(j);
          final double logP = Math.min(logQ + logAlpha, 0d);
          final D object = nonZeroObjects.get(j);
          logPValues.add(logP);
          if (logP == 0d) {
            keeperLogWeights.add(logQ);
            keeperObjects.add(object);
          } else {
            belowObjects.add(object);
            belowPTotal = ExtLogMath.add(belowPTotal, logQ);
            belowLogWeights.add(logQ);
          }
        }

        if (keeperLogWeights.isEmpty()) {
          /*
           * All weights are below, 
           * resample without replacement (and use current weights)
           * resample with replacement,
           * take top N (and use current weights)?
           */
//          resultObjects = sampleNoReplaceMultipleLogScaleES(Doubles.toArray(nonZeroLogWeights), 
//              nonZeroObjects, random, N);
//          resultObjects = sampleReplaceCumulativeLogScale(
//              Doubles.toArray(cumNonZeroLogWeights), nonZeroObjects, random, N);
          resultObjects = Lists.newArrayList(lowVarianceSampler(nonZeroDist, random, N));
          resultLogWeights = Collections.nCopies(N, -Math.log(N));
          log.debug("all below logAlpha");
        } else {
          wasWaterFillingApplied = true;
          log.debug("water-filling applied!");
          if (!belowLogWeights.isEmpty()) {
            /*
             * Resample the below beta entries
             */
            final int resampleN = N - keeperLogWeights.size();
            List<D> belowObjectsResampled = sampleNoReplaceMultipleLogScaleES(Doubles.toArray(belowLogWeights), 
                belowObjects, random, resampleN);
            List<Double> belowWeightsResampled = Collections.nCopies(resampleN, -logAlpha);
            
            keeperObjects.addAll(belowObjectsResampled);
            keeperLogWeights.addAll(belowWeightsResampled);
          } 
          
          assert isLogNormalized(keeperLogWeights, 1e-7);

          resultObjects = keeperObjects;
          resultLogWeights = keeperLogWeights;
        } 
      }
    }
    
    Preconditions.checkState(resultLogWeights.size() == resultObjects.size()
        && resultLogWeights.size() == N);
    WFCountedDataDistribution<D> result = new WFCountedDataDistribution<D>(N, true);
    result.setWasWaterFillingApplied(wasWaterFillingApplied);
    for (int i = 0; i < N; i++) {
      result.increment(resultObjects.get(i), resultLogWeights.get(i));
    }
    return result;
  }

  /**
   * Checks that weights are normalized up to the given magnitude.
   * 
   * @param logWeights
   * @param zeroPrec
   * @return
   */
  public static boolean
      isLogNormalized(final Collection<Double> logWeights, final double zeroPrec) {
    return isLogNormalized(Doubles.toArray(logWeights), zeroPrec);
  }

  /**
   * Checks that weights are normalized up to the given magnitude.
   * 
   * @param logWeights
   * @param zeroPrec
   * @return
   */
  public static boolean
      isLogNormalized(final double[] logWeights, final double zeroPrec) {
    Preconditions.checkArgument(zeroPrec > 0d && zeroPrec < 1e-3);
    double logTotal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < logWeights.length; i++) {
      logTotal = ExtLogMath.add(logTotal, logWeights[i]);
    }
    return Math.abs(logTotal) < zeroPrec;
  }

  /**
   * Find the log of alpha: the water-filling cut-off.
   * The logTotalWeight is important for numerical stability.  Since
   * we're looping through and subtracting log values, we need
   * to know what "zero" is (relative to the numeric precision). 
   * <br>
   * Note: the log weights must already be normalized.  Even though
   * the total must be provided, no normalization happens here.
   * <br>
   * TODO: should get rid of the total weight, or normalize when not, but
   * I'd rather get rid of the isLogNormalized check...
   * 
   * @param logWeights normalized log weights
   * @param logTotalWeight log weights sum
   * @param N
   * @return
   */
  public static double findLogAlpha(final double[] logWeights, final double logTotalWeight, final int N) {
    final int M = logWeights.length;
    final double[] sLogWeights = logWeights.clone();
    Arrays.sort(sLogWeights);
    ArrayUtil.reverse(sLogWeights);
    double logTailSum = Math.abs(logTotalWeight);
    double logAlpha = Math.log(N);
    int k = 0;
    int pk = k;
    
    assert ExtSamplingUtils.isLogNormalized(sLogWeights, 1e-7);

    while (true) {
      pk = k;
      while (k < M && logAlpha + sLogWeights[k] > 0) {
        final double thisLogWeight = sLogWeights[k];
        final double logTailSumTmp;
        /*
         * TODO FIXME XXX: terrible hack!  fix this!
         */
        if (logTailSum < thisLogWeight) {
          final double tmpDiff = Math.exp(logTailSum) - Math.exp(thisLogWeight);
          if (Math.abs(tmpDiff) < 1e-1) {
            logTailSumTmp = Double.NEGATIVE_INFINITY;
            log.warn("numerical instability");
          } else {
            logTailSumTmp = ExtLogMath.subtract(logTailSum, thisLogWeight);
          }
        } else {
          logTailSumTmp = ExtLogMath.subtract(logTailSum, thisLogWeight);
        }

        Preconditions.checkState(!Double.isNaN(logTailSumTmp));
        logTailSum = logTailSumTmp;
        k++;
      }
      logAlpha = Math.log(N-k) - logTailSum;
      if ( pk == k || k == M ) 
        break;
    }
    
//    /*
//     * Here's a special case when we have N-many disproportionately 
//     * large and equal weights...we just reset alpha so that resampling
//     * is required.  This wouldn't be good if we wanted to know something
//     * more about our weights, but we don't need to, yet. 
//     */
//    if (k == N && Double.isNaN(logAlpha))
//      return logAlpha;
//    else
//      Preconditions.checkState(!Double.isNaN(logAlpha));
  
    return logAlpha;
  }
  
  public static int sampleIndexFromLogProbabilities(final Random random, final double[] logProbs,
      double totalLogProbs) {
    double value = Math.log(random.nextDouble());
    final int lastIndex = logProbs.length - 1;
    for (int i = 0; i < lastIndex; i++) {
      value = ExtLogMath.subtract(value, logProbs[i] - totalLogProbs);
      if (Double.isNaN(value) || value == Double.NEGATIVE_INFINITY) {
        return i;
      }
    }
    return lastIndex;
  }
  
  /**
   * Sort a[] into descending order by "heapsort";
   * sort ib[] alongside;
   * if initially, ib[] = 1...n, it will contain the permutation finally
   *
   * From R's <a href="https://github.com/wch/r-source/blob/2633f0ee6306b23eadda989110f5748ce4c050bd/src/main/sort.c#L264">source code</a>.
   * @param a
   * @param ib
   * @param n
   */
  public static void revsort(double[] a, int[] ib, int n) {
    /*
     * Sort a[] into descending order by "heapsort"; sort ib[] alongside; if
     * initially, ib[] = 1...n, it will contain the permutation finally
     */

    int l, j, ir, i;
    double ra;
    int ii;

    if (n <= 1)
      return;

//    a--;
//    ib--;

    l = (n >> 1) + 1;
    ir = n;

    for (;;) {
      if (l > 1) {
        l = l - 1;
        ra = a[l - 1];
        ii = ib[l - 1];
      } else {
        ra = a[ir - 1];
        ii = ib[ir - 1];
        a[ir - 1] = a[0];
        ib[ir - 1] = ib[0];
        if (--ir == 1) {
          a[0] = ra;
          ib[0] = ii;
          return;
        }
      }
      i = l;
      j = l << 1;
      while (j <= ir) {
        if (j < ir && a[j - 1] > a[j])
          ++j;
        if (ra > a[j - 1]) {
          a[i - 1] = a[j - 1];
          ib[i - 1] = ib[j - 1];
          j += (i = j);
        } else
          j = ir + 1;
      }
      a[i - 1] = ra;
      ib[i - 1] = ii;
    }
  }

  /**
   * Weighed sampling without replacement from R's 
   * <a href="https://github.com/wch/r-source/blob/trunk/src/main/random.c">source</a>.
   * 
   * @param n
   * @param p
   * @param perm
   * @param nans
   * @param ans
   * @param rng
   */
  public static void probSampleNoReplace(int n, double[] p, int[] perm, int nans,
      int[] ans, Random rng) {
    double rT, mass, totalmass;
    int i, j, k, n1;

    /* Record element identities */
    for (i = 0; i < n; i++)
      perm[i] = i + 1;

    /* Sort probabilities into descending order */
    /* Order element identities in parallel */
    revsort(p, perm, n);

    /* Compute the sample */
    totalmass = 1d;
    for (i = 0, n1 = n - 1; i < nans; i++, n1--) {
      rT = totalmass * rng.nextDouble();
      mass = 0d;
      for (j = 0; j < n1; j++) {
        mass += p[j];
        if (rT <= mass)
          break;
      }
      ans[i] = perm[j];
      totalmass -= p[j];
      for (k = j; k < n1; k++) {
        p[k] = p[k + 1];
        perm[k] = perm[k + 1];
      }
    }
  }

//   * <a href="https://www.sciencedirect.com/science/article/pii/S002001900500298X">
//   *  Pavlos S. Efraimidis, Paul G. Spirakis, Weighted random sampling with a reservoir</a>
//   *  from the discussion 
//   *  <a href="http://stackoverflow.com/questions/15113650/faster-weighted-sampling-without-replacement">
//   *  here</a>
  /**
   * Sample without replacement for a given support/domain and log weights.
   * 
   * @see ExtSamplingUtils#probSampleNoReplace(int, double[], int[], int, int[], Random) 
   *  
   * @param logWeights
   * @param logWeightSum
   * @param domain
   * @param random
   * @param numSamples
   * @return
   */
  public static <D> List<D> sampleNoReplaceMultipleLogScale(final double[] logWeights,
      final double logWeightSum, final Collection<D> domain, 
      final Random random, final int numSamples) {

    Preconditions.checkArgument(domain.size() >= numSamples, 
        "domain size must be >= numSamples");
    
    if (domain.size() == numSamples) {
      return Lists.newArrayList(domain);
    }
    
    double weightSum = 0d;
    double[] weights = new double[logWeights.length];
    for (int i = 0; i < weights.length; i++) {
      weights[i] = Math.exp(logWeights[i] - logWeightSum);
      weightSum += weights[i];
    }
    Preconditions.checkState(Math.abs(weightSum - 1d) < 1e-5);
    int[] perm = new int[logWeights.length];
    int[] ans = new int[numSamples];
    probSampleNoReplace(logWeights.length, weights, perm, numSamples, ans, random);
    final List<D> samples = Lists.newArrayListWithCapacity(numSamples);
    for (int i = 0; i < ans.length; i++) {
      samples.add(Iterables.get(domain, ans[i]-1));
    }
    
    return samples;
  }

  /**
   * Streaming weighted sampling without replacement.
   * 
   * @param logWeights
   * @param domain
   * @param random
   * @param numSamples
   * @return
   */
  public static <D> List<D> sampleNoReplaceMultipleLogScaleES(final double[] logWeights,
      final Collection<D> domain, final Random random, final int numSamples) {

    Preconditions.checkArgument(domain.size() >= numSamples, 
        "domain size must be >= numSamples");
    
    if (domain.size() == numSamples) {
      return Lists.newArrayList(domain);
    }

    MinMaxPriorityQueue<ExtDefaultWeightedValue<D>> pQueue = MinMaxPriorityQueue
        .orderedBy(Ordering.natural().reverse())
        .maximumSize(numSamples).create();

    for (int i = 0; i < logWeights.length; i++) {
      final double sampleKey = Math.pow(random.nextDouble(), 
          1d/Math.exp(logWeights[i]));
      pQueue.add(ExtDefaultWeightedValue.create(
          Iterables.get(domain, i), sampleKey));
    }

    final List<D> samples = Lists.newArrayListWithCapacity(numSamples);
    while(samples.size() < numSamples) {
      samples.add(pQueue.poll().getValue());
    }
    
    return samples;
  }

  /**
   * Streaming weighted sampling without replacement using exponential jumps.
   * 
   * @param logWeights
   * @param domain
   * @param random
   * @param numSamples
   * @return
   */
  public static <D> List<D> sampleNoReplaceMultipleLogScaleStreamES(
      final double[] logWeights,
      final Collection<D> domain, final Random random, final int numSamples) {

    Preconditions.checkArgument(domain.size() >= numSamples, 
        "domain size must be >= numSamples");
    
    if (domain.size() == numSamples) {
      return Lists.newArrayList(domain);
    }

    TreeMap<Double, D> tMap = Maps.<Double, Double, D>newTreeMap(new Comparator<Double>() {
      @Override
      public int compare(Double o1, Double o2) {
        return (o1 > o2) ? -1 : 1;
      }
    });

    double r1 = 0d, expJump = 0d, currentWeight = 0d, nextWeight;
    boolean inFlight = false;
    long itemsProcessed = 0;
    for (int i = 0; i < logWeights.length; i++) {
      Entry<Double, D> rWorstItem;
      double currentThreshold;

      final D thisItem = Iterables.get(domain, i);
      final double thisWeight = Math.exp(logWeights[i]);
      if (itemsProcessed < numSamples) {
        final double newWeight = Math.pow(random.nextDouble(), 1d/thisWeight);
        tMap.put(newWeight, thisItem);
      } else {
  			rWorstItem = tMap.firstEntry();
  			currentThreshold = rWorstItem.getKey();
        if (!inFlight) {
  				// Generate exponential jump
  				r1 = random.nextDouble();
  				expJump = Math.log(r1) / Math.log(currentThreshold);
  				
  				currentWeight = 0d;
  				nextWeight = 0d;
  				
  				inFlight = true;
          
        } else {
    			// Check if the Exponential Jump lands on the current item
    			nextWeight = currentWeight + thisWeight;
    			if (expJump < nextWeight) {
    				double lowJ = currentWeight;
    				double highJ = nextWeight;
    				
    				// We have to calculate a key for the new item
    				// The key has to be in the interval (key-of-replaced-item, max-key]
    				double lowR = Math.pow(currentThreshold, thisWeight);
    
    				// We use the random number of the exponential jump 
    				// to calculate the random key
    				// The random number has to be "normalized" for its new use
    				double lthr = Math.pow(currentThreshold, highJ);
    				double hthr = Math.pow(currentThreshold, lowJ);
    				double r2 = (r1 - lthr) / (hthr - lthr);
    
    				// OK double r3 = lowR + (1-lowR) * myRandom.rand();
    				double r3 = lowR + (1 - lowR) * r2; // myRandom.rand();
    				double key = Math.pow(r3, 1 / thisWeight);
    
    				// Insert the Item into the Reservoir
    				tMap.put(key, thisItem);
    				
    				inFlight = false;
    			} else {
    				currentWeight = nextWeight;
    			}	
        }
    		itemsProcessed++;
      }
    }
    
    final List<D> samples = Lists.newArrayListWithCapacity(numSamples);
    while(samples.size() < numSamples) {
      samples.add(tMap.pollFirstEntry().getValue());
    }
    
    return samples;
  }

  
  /**
   * Sample with replacement for cumulative log weights.
   * NOTE: The given weights MUST be cumulative (in increasing order)!
   *  
   * @param cumulativeLogWeights
   * @param logWeightSum
   * @param domain
   * @param random
   * @param numSamples
   * @return
   */
  public static <D> List<D> sampleReplaceCumulativeLogScale(
      final double[] cumulativeLogWeights, 
      final Collection<D> domain, 
      final Random random, final int numSamples) {
    Preconditions.checkArgument(domain.size() == cumulativeLogWeights.length,
        "domain size must be equala to number of cumulative log weights");

    final double logWeightSum = cumulativeLogWeights[cumulativeLogWeights.length-1];
    final List<D> samples = Lists.newArrayListWithCapacity(numSamples);
    int index;
    for (int n = 0; n < numSamples; n++) {
      final double p = logWeightSum + Math.log(random.nextDouble());
      index = Arrays.binarySearch(cumulativeLogWeights, p);
      if (index < 0) {
        final int insertionPoint = -index - 1;
        index = insertionPoint;
      }
      D sampledObj = Iterables.get(domain, index);
      samples.add(sampledObj);
    }
    return samples;

  }

  public static double logSum(final double[] logWeights) {
    double pTotal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < logWeights.length; i++) {
      pTotal =
          ExtLogMath.add(pTotal, logWeights[i]);
    }
    return pTotal;
  }

  public static void logNormalize(final double[] logWeights) {
    final double totalLogWeights = ExtSamplingUtils.logSum(
        logWeights);
    for (int i = 0; i < logWeights.length; i++) {
      logWeights[i] -= totalLogWeights;
    }
  }

  public static double[] accumulate(Collection<Double> logLikelihoods) {
    double pTotal = Double.NEGATIVE_INFINITY;
    double[] result = new double[logLikelihoods.size()];
    int i = 0;
    for (double lik : logLikelihoods) {
      pTotal =
          ExtLogMath.add(pTotal, lik);
      result[i] = pTotal;
      i++;
    }
    return result;
  }

  /**
   * Resample log-weighed objects with replacement.
   * 
   * @param cumulativeWeightedObjs
   * @param maxLogWeight
   * @param random
   * @param numParticles
   * @return
   */
  public static <D extends ComparableWeighted> List<D> sampleReplaceCumulativeLogScale(
      List<D> cumulativeWeightedObjs, Random random,
      int numSamples) {

    final List<D> samples = Lists.newArrayListWithCapacity(numSamples);
    int index;
    final double totalLogWeight = Iterables.getLast(cumulativeWeightedObjs).getWeight();
    for (int n = 0; n < numSamples; n++) {
      final double p = totalLogWeight + Math.log(random.nextDouble());
      index = Collections.binarySearch(cumulativeWeightedObjs, 
          DefaultWeightedValue.create(null, p));
      if (index < 0) {
        final int insertionPoint = -index - 1;
        index = insertionPoint;
      }
      final D sampledObj = Iterables.get(cumulativeWeightedObjs, index);
      samples.add(sampledObj);
    }
    return samples;
  }

}
