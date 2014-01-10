package com.statslibextensions.statistics;

import static org.junit.Assert.*;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.distribution.DefaultDataDistribution;
import gov.sandia.cognition.statistics.distribution.PoissonDistribution;
import gov.sandia.cognition.util.Pair;
import gov.sandia.cognition.util.WeightedValue;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.TreeSet;

import org.junit.Test;

import com.google.common.collect.DiscreteDomains;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multiset;
import com.google.common.collect.Range;
import com.google.common.collect.Ranges;
import com.google.common.primitives.Doubles;
import com.statslibextensions.math.ExtLogMath;
import com.statslibextensions.math.MutableDoubleCount;
import com.statslibextensions.statistics.distribution.CountedDataDistribution;
import com.statslibextensions.statistics.ExtSamplingUtils;
import com.statslibextensions.util.ExtStatisticsUtils;

public class ExtSamplingUtilsTest {

  @Test
  public void testFindLogAlpha1() {
    double[] testLogWeights =
        new double[] { Math.log(5d / 11d), Math.log(3d / 11d),
            Math.log(2d / 11d), Math.log(1d / 11d) };

    final double logAlpha1 =
        ExtSamplingUtils.findLogAlpha(testLogWeights, 0d, 1);

    double pTotal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < testLogWeights.length; i++) {
      pTotal =
          ExtLogMath.add(pTotal,
              Math.min(testLogWeights[i] + logAlpha1, 0d));
    }
    assertEquals(Math.log(1), pTotal, 1e-7);
    assertEquals(0, logAlpha1, 1e-7);

    final double logAlpha2 =
        ExtSamplingUtils.findLogAlpha(testLogWeights, 0d, 2);

    pTotal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < testLogWeights.length; i++) {
      pTotal =
          ExtLogMath.add(pTotal,
              Math.min(testLogWeights[i] + logAlpha2, 0d));
    }
    assertEquals(Math.log(2), pTotal, 1e-7);
    assertEquals(0.6931471805599d, logAlpha2, 1e-7);

    final double logAlpha3 =
        ExtSamplingUtils.findLogAlpha(testLogWeights, 0d, 3);

    pTotal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < testLogWeights.length; i++) {
      pTotal =
          ExtLogMath.add(pTotal,
              Math.min(testLogWeights[i] + logAlpha3, 0d));
    }
    assertEquals(Math.log(3), pTotal, 1e-7);
    assertEquals(1.29928298413d, logAlpha3, 1e-7);
  }
  
  @Test
  public void testFindLogAlpha2() {
    double[] testLogWeights =
        new double[] { 
        -4.758895070648572, -2.561670493312353, 
        -4.954271557120856, -2.757046979784638, 
        -2.757046979784638, -4.954271557120856, 
        -4.954271557120856, -2.757046979784638, 
        -2.757046979784638, -4.954271557120856 };

    ExtSamplingUtils.logNormalize(testLogWeights);

    final double logAlpha1 =
        ExtSamplingUtils.findLogAlpha(testLogWeights, 0d, 5);

    double pTotal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < testLogWeights.length; i++) {
      pTotal =
          ExtLogMath.add(pTotal,
              Math.min(testLogWeights[i] + logAlpha1, 0d));
    }
    assertEquals(Math.log(5), pTotal, 1e-7);
  }

  /**
   * 10 weights all about 1/10, then 10 are about zero.  Since
   * we can't really water-fill at this high a precision, we should
   * allow it to resample, which would give the same results (given
   * that there are ten weights that are significantly greater than
   * the other ten, we'll end up taking those regardless).  
   */
  @Test
  public void testFindLogAlpha3() {
//    double[] testLogWeights2 = { 
//        -2.3025850929940397, -2.3025850929940397, 
//        -2.3025850929940397, -2.3025850929940397, -2.3025850929940397, 
//        -2.3025850929940397, -2.3025850929940397, -2.3025850929940397, 
//        -2.3025850929940397, -2.3025850929940397, -341.22111709254136, 
//        -341.22111709254136, -341.22111709254136, -341.22111709254136, 
//        -341.22111709254136, -341.22111709254136, -341.22111709254136, 
//        -341.22111709254136, -341.22111709254136, -341.22111709254136 }; 
//
//    double s1 = 0d;
//    for (int i = 0; i < testLogWeights2.length; i++) {
//      s1 = LogMath2.subtract(s1, testLogWeights2[i]);
//    }

    double[] testLogWeights =
        new double[] { 
        -341.22111709254136, -2.3025850929940397, -341.22111709254136, 
         -2.3025850929940397, -341.22111709254136, -2.3025850929940397, 
         -341.22111709254136, -2.3025850929940397, -341.22111709254136, 
         -2.3025850929940397, -341.22111709254136, -2.3025850929940397, 
         -341.22111709254136, -2.3025850929940397, -341.22111709254136, 
         -2.3025850929940397, -341.22111709254136, -2.3025850929940397, 
         -341.22111709254136, -2.3025850929940397 };

    final double totalLogWeight = 5.995204332975845E-15;

//    SamplingUtils.logNormalize(testLogWeights);

    final double logAlpha1 =
        ExtSamplingUtils.findLogAlpha(testLogWeights, totalLogWeight, 10);

    assertTrue(Double.isInfinite(logAlpha1));
  }

  /**
   * Test basic resample with flat weights
   */
  @Test
  public void testWaterFillingResample1() {
    double[] testLogWeights =
        new double[] { Math.log(1d / 4d), Math.log(1d / 4d),
            Math.log(1d / 4d), Math.log(1d / 4d) };
    String[] testObjects = new String[] { "o1", "o2", "o3", "o4" };

    final Random rng = new Random(123569869l);
    final int N = 2;
    DataDistribution<String> wfResampleResults =
        ExtSamplingUtils.waterFillingResample(testLogWeights, 0d,
            Lists.newArrayList(testObjects), rng, N);

    for (Entry<String, ? extends Number> logWeight : wfResampleResults
        .asMap().entrySet()) {
      assertEquals(-Math.log(N), logWeight.getValue().doubleValue(),
          1e-7);
    }
  }

  /**
   * Test error when < N non-zero weights
   */
  @Test(expected = IllegalStateException.class)
  public void testWaterFillingResample2() {
    double[] testLogWeights =
        new double[] { Double.NEGATIVE_INFINITY,
            Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, 0d };
    String[] testObjects = new String[] { "o1", "o2", "o3", "o4" };

    final Random rng = new Random(123569869l);
    final int N = 2;
    DataDistribution<String> wfResampleResults =
        ExtSamplingUtils.waterFillingResample(testLogWeights, 0d,
            Lists.newArrayList(testObjects), rng, N);

    assertEquals(0d, wfResampleResults.getMaxValue(), 1e-7);
    assertEquals("o4", wfResampleResults.getMaxValueKey());
    final int count =
        ((MutableDoubleCount) wfResampleResults.asMap().get(
            wfResampleResults.getMaxValueKey())).count;
    assertEquals(2, count);
  }

  /**
   * Test basic resample with N non-zero weights TODO: really just checking for
   * flat weights now, need to check more?
   */
  @Test
  public void testWaterFillingResample3() {
    double[] testLogWeights =
        new double[] { Double.NEGATIVE_INFINITY,
            Double.NEGATIVE_INFINITY, Math.log(1d / 2d),
            Math.log(1d / 2d) };
    String[] testObjects = new String[] { "o1", "o2", "o3", "o4" };

    final Random rng = new Random(123569869l);
    final int N = 2;
    DataDistribution<String> wfResampleResults =
        ExtSamplingUtils.waterFillingResample(testLogWeights, 0d,
            Lists.newArrayList(testObjects), rng, N);

    for (Entry<String, ? extends Number> logWeight : wfResampleResults
        .asMap().entrySet()) {
      assertEquals(-Math.log(N), logWeight.getValue().doubleValue(),
          1e-7);
    }
  }

  /**
   * Test water-filling accepts one and resamples the others
   */
  @Test
  public void testWaterFillingResample4() {
    double[] testLogWeights =
        new double[] { Math.log(5d / 11d), Math.log(3d / 11d),
            Math.log(2d / 11d), Math.log(1d / 11d) };
    String[] testObjects = new String[] { "o1", "o2", "o3", "o4" };

    final Random rng = new Random(123569869l);
    final int N = 3;
    DataDistribution<String> wfResampleResults =
        ExtSamplingUtils.waterFillingResample(testLogWeights, 0d,
            Lists.newArrayList(testObjects), rng, N);

    assertEquals(Math.log(5d / 11d), wfResampleResults.getMaxValue(),
        1e-7);
    assertEquals("o1", wfResampleResults.getMaxValueKey());

    final double logAlpha =
        ExtSamplingUtils.findLogAlpha(testLogWeights, 0d, N);

    List<Double> logWeights = Lists.newArrayList();
    for (Entry<String, ? extends Number> entry : wfResampleResults
        .asMap().entrySet()) {
      final double logWeight = entry.getValue().doubleValue();
      logWeights.add(logWeight);
      if (!entry.getKey().equals(wfResampleResults.getMaxValueKey())) {
        assertEquals(-logAlpha, logWeight, 1e-7);
      }
    }

    assertTrue(ExtSamplingUtils.isLogNormalized(logWeights, 1e-7));
  }

  /**
   * Test water-filling accepts two and resamples the others
   */
  @Test
  public void testWaterFillingResample5() {
    double[] testLogWeights =
        new double[] { Math.log(6d / 17d), Math.log(5d / 17d),
            Math.log(3d / 17d), Math.log(2d / 17d),
            Math.log(1d / 17d) };
    String[] testObjects =
        new String[] { "o0", "o1", "o2", "o3", "o4" };

    final Random rng = new Random(123569869l);
    final int N = 4;
    DataDistribution<String> wfResampleResults =
        ExtSamplingUtils.waterFillingResample(testLogWeights, 0d,
            Lists.newArrayList(testObjects), rng, N);

    assertEquals(Math.log(6d / 17d), wfResampleResults.getMaxValue(),
        1e-7);
    assertEquals("o0", wfResampleResults.getMaxValueKey());
    DataDistribution<String> tmpResults = wfResampleResults.clone();
    tmpResults.decrement(wfResampleResults.getMaxValueKey(),
        tmpResults.getMaxValue());
    assertEquals(Math.log(5d / 17d), tmpResults.getMaxValue(), 1e-7);
    assertEquals("o1", tmpResults.getMaxValueKey());

    final double logAlpha =
        ExtSamplingUtils.findLogAlpha(testLogWeights, 0d, N);

    double pTotal = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < testLogWeights.length; i++) {
      pTotal =
          ExtLogMath.add(pTotal,
              Math.min(testLogWeights[i] + logAlpha, 0d));
    }
    assertEquals(Math.log(N), pTotal, 1e-7);
    assertEquals(1.734601055388d, logAlpha, 1e-7);

    List<Double> logWeights = Lists.newArrayList();
    int i = 0;
    for (Entry<String, ? extends Number> entry : wfResampleResults
        .asMap().entrySet()) {
      final double logWeight = entry.getValue().doubleValue();
      logWeights.add(logWeight);
      if (i > 1) {
        assertEquals(-logAlpha, logWeight, 1e-7);
      }
      i++;
    }

    assertTrue(ExtSamplingUtils.isLogNormalized(logWeights, 1e-7));
  }
  
  /**
   * Test resampling with replacement for num samples >> support size.
   * Weights are deterministic.
   */
  @Test
  public void testResampleWithReplace1() {
    double[] logWeights = {Math.log(6d/10d), Math.log(2d/10d), Math.log(1d/10d), Math.log(1d/10d)};
    double[] cumulLogWeights = ExtSamplingUtils.accumulate(Doubles.asList(logWeights));
    List<String> domain = Lists.newArrayList("e1", "e2", "e3", "e4");
    Random random = new Random();
    final int numSamples = 200000;
    final int R = 1;

    CountedDataDistribution<String> sampleDist = new CountedDataDistribution<String>(false);
    for (int i = 0; i < R; i++) {
      List<String> result = ExtSamplingUtils.sampleReplaceCumulativeLogScale(
          cumulLogWeights, domain, random, numSamples);
      sampleDist.incrementAll(result);
    }
    
    int i = 0;
    for (String el : domain) {
      System.out.println(el + "=" + sampleDist.getFraction(el) + "(" + Math.exp(logWeights[i]) + ")");
      assertEquals(Math.exp(logWeights[i]), 
          sampleDist.getFraction(el), 1e-2);
      i++;
    }
    
  }

  /**
   * Test resampling with replacement for num samples >> support size.
   * Randomly determines weights for test
   */
  @Test
  public void testResampleWithReplace2() {
    Random random = new Random();
    final int N = 10;
    List<Double> logWeights = Lists.newArrayList();
    List<String> domain = Lists.newArrayList();
    DataDistribution<String> initialDist = new DefaultDataDistribution<String>();
//        new CountedDataDistribution<String>(false);
    while (initialDist.getDomainSize() < 10) {
      final double smple = Math.log(random.nextDouble());
      logWeights.add(smple);
      String obj = "obj" + random.nextInt(N);
      domain.add(obj);
      initialDist.increment(obj, Math.exp(smple));
    }
    System.out.println(ExtStatisticsUtils.prettyPrintDistribution(initialDist));

    double[] cumulLogWeights = ExtSamplingUtils.accumulate(logWeights);
    final int numSamples = 200000;
    final int R = 1;

    /*
     * numSamples >> N
     */
    CountedDataDistribution<String> sampleDist = new CountedDataDistribution<String>(false);
    for (int i = 0; i < R; i++) {
      List<String> result = 
//          (List<String>) initialDist.sample(random, numSamples);
          ExtSamplingUtils.sampleReplaceCumulativeLogScale(
          cumulLogWeights, domain, random, numSamples);
      sampleDist.incrementAll(result);
    }
    
    for (String el : domain) {
      System.out.println(el + "=" + sampleDist.getFraction(el) 
          + "(" + initialDist.getFraction(el) + ")");
      assertEquals(initialDist.getFraction(el), sampleDist.getFraction(el), 1e-2);
    }
  }

  /**
   * Test resampling with replacement for num samples < support size.
   * Randomly determines weights for test
   */
  @Test
  public void testResampleWithReplace3() {
    Random random = new Random();
    final int N = 100;
    List<Double> logWeights = Lists.newArrayList();
    List<String> domain = Lists.newArrayList();
    DataDistribution<String> initialDist = new DefaultDataDistribution<String>();
//        new CountedDataDistribution<String>(false);
    while (initialDist.getDomainSize() < N) {
      final double smple = Math.log(random.nextDouble());
      logWeights.add(smple);
      String obj = "obj" + random.nextInt(N);
      domain.add(obj);
      initialDist.increment(obj, Math.exp(smple));
    }
    System.out.println(ExtStatisticsUtils.prettyPrintDistribution(initialDist));

    double[] cumulLogWeights = ExtSamplingUtils.accumulate(logWeights);
    /*
     * Now, do the same for numSamples < N
     */
    final int M = 10;
    final int R = 50000;
    CountedDataDistribution<String> sampleDist = new CountedDataDistribution<String>(false);
    for (int i = 0; i < R; i++) {
      List<String> result = ExtSamplingUtils.sampleReplaceCumulativeLogScale(
        cumulLogWeights, domain, random, M);
      sampleDist.incrementAll(result);
    }
    
    for (String el : initialDist.getDomain()) {
      final double obsFreq = sampleDist.getCount(el)/(double)R;
      final double expFreq = M * initialDist.getFraction(el);
      System.out.println(el + ":\tfreq=" + obsFreq + "\t(" + expFreq  + ")");
      assertEquals(expFreq, obsFreq, 1e-2);
    }
    
  }

  /**
   * Sample one from the no-replace sampler and make sure
   * it gives a good sample distribution, just for a sanity check. 
   */
  @Test
  public void testResampleNoReplace1() {
    double[] logWeights = {Math.log(6d/10d), Math.log(2d/10d), Math.log(1d/10d), Math.log(1d/10d)};
    List<String> domain = Lists.newArrayList("e1", "e2", "e3", "e4");
    Random random = new Random();
    final int numSamples = 1;
    final int R = 3000000;

    CountedDataDistribution<String> sampleDist = new CountedDataDistribution<String>(false);
    for (int i = 0; i < R; i++) {
      List<String> result = ExtSamplingUtils.sampleNoReplaceMultipleLogScale(
          logWeights, 0d, domain, random, numSamples);
      sampleDist.incrementAll(result);
    }
    
    int i = 0;
    for (String el : domain) {
      System.out.println(el + "=" + sampleDist.getFraction(el) + "(" + Math.exp(logWeights[i]) + ")");
      assertEquals(logWeights[i], sampleDist.getLogFraction(el), 1e-2);
      i++;
    }
    
  }

  /**
   * Sample one from the no-replace sampler and make sure
   * it gives a good sample distribution, just for a sanity check. 
   */
  @Test
  public void testResampleNoReplaceES1() {
    double[] logWeights = {Math.log(6d/10d), Math.log(2d/10d), Math.log(1d/10d), Math.log(1d/10d)};
    List<String> domain = Lists.newArrayList("e1", "e2", "e3", "e4");
    Random random = new Random();
    final int numSamples = 1;
    final int R = 3000000;

    CountedDataDistribution<String> sampleDist = new CountedDataDistribution<String>(false);
    for (int i = 0; i < R; i++) {
      List<String> result = ExtSamplingUtils.sampleNoReplaceMultipleLogScaleES(
          logWeights, domain, random, numSamples);
      sampleDist.incrementAll(result);
    }
    
    int i = 0;
    for (String el : domain) {
      System.out.println(el + "=" + sampleDist.getFraction(el) + "(" + Math.exp(logWeights[i]) + ")");
      assertEquals(logWeights[i], sampleDist.getLogFraction(el), 1e-2);
      i++;
    }
    
  }

  /**
   * Test the order distributions, i.e. check the observed frequencies
   * for the first, second,... sampled particles.
   */
//  @Test
  public void testResampleNoReplaceES2() {
    Random random = new Random();
    final int N = 10;
    List<Double> logWeights = Lists.newArrayList();
    List<String> domain = Lists.newArrayList();
    DataDistribution<String> initialDist = new DefaultDataDistribution<String>();
    while (initialDist.getDomainSize() < N) {
      final double smple = Math.log(random.nextDouble());
      logWeights.add(smple);
      String obj = "obj" + random.nextInt(N);
      domain.add(obj);
      initialDist.increment(obj, Math.exp(smple));
    }
    System.out.println(ExtStatisticsUtils.prettyPrintDistribution(initialDist));
    final int R = 50000;

    List<Map<String, Double>> expectedOrderMeans = Lists.newArrayList();
    List<DataDistribution<String>> orderStats = Lists.newArrayList();
    for (int k = 0; k < N; k++) {
      DataDistribution<String> sampleDist = new CountedDataDistribution<String>(false);
      orderStats.add(sampleDist);
      // TODO FIXME need to compute combinations and their probabilities
    }

    for (int i = 0; i < R; i++) {
      List<String> result = ExtSamplingUtils.sampleNoReplaceMultipleLogScaleES(
         Doubles.toArray(logWeights), domain, random, logWeights.size());
      for (int k = 0; k < N; k++) {
        orderStats.get(k).increment(result.get(k));
      }
    }
    
    for (int k = 0; k < N; k++) {
      for (String el : initialDist.getDomain()) {
        final double obsFreq = orderStats.get(k).getFraction(el);
        final double expFreq = expectedOrderMeans.get(k).get(el);
        System.out.println(el + ":\tfreq=" + obsFreq + "\t(" + expFreq  + ")");
        assertEquals(expFreq, obsFreq, 1e-2);
      }

    }
  }

  /**
   * Test resampling with replacement (systematic sampler) 
   * for num samples >> support size.
   * Randomly determines weights for test
   */
  @Test
  public void testLowVarResampleWithReplace1() {
    Random random = new Random();
    final int N = 10;
    List<Double> logWeights = Lists.newArrayList();
    List<String> domain = Lists.newArrayList();
    DataDistribution<String> initialDist = new DefaultDataDistribution<String>();
    while (initialDist.getDomainSize() < 10) {
      final double smple = Math.log(random.nextDouble());
      logWeights.add(smple);
      String obj = "obj" + random.nextInt(N);
      domain.add(obj);
      initialDist.increment(obj, Math.exp(smple));
    }
    System.out.println(ExtStatisticsUtils.prettyPrintDistribution(initialDist));

    final int numSamples = 200000;
    final int R = 1;

    /*
     * numSamples >> N
     */
    CountedDataDistribution<String> sampleDist = new CountedDataDistribution<String>(false);
    TreeSet<WeightedValue<String>> wvalues = ExtSamplingUtils.getLogWeighedList(initialDist);
    for (int i = 0; i < R; i++) {
      Multiset<String> result = 
          ExtSamplingUtils.lowVarianceSampler(wvalues, random, numSamples);
      sampleDist.incrementAll(result);
    }
    
    for (String el : domain) {
      System.out.println(el + "=" + sampleDist.getFraction(el) 
          + "(" + initialDist.getFraction(el) + ")");
      assertEquals(initialDist.getFraction(el), sampleDist.getFraction(el), 1e-2);
    }
  }
}
