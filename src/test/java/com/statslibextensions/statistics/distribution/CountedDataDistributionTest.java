package com.statslibextensions.statistics.distribution;

import java.util.Random;

import org.junit.Assert;
import org.junit.Test;

import com.statslibextensions.statistics.distribution.CountedDataDistribution;

public class CountedDataDistributionTest {

  @Test
  public void testCountedDataDistributionLogScale() {
    final CountedDataDistribution<String> testDist =
        new CountedDataDistribution<String>(true);
    Assert.assertTrue(testDist.isLogScale());
    Assert.assertTrue(testDist.isEmpty());

    testDist.set("item1", Math.log(10d), 1);
    testDist.set("item2", Math.log(20d), 1);
    testDist.set("item3", Math.log(30d), 1);
    testDist.set("item4", Math.log(40d), 1);

    Assert.assertEquals(1, testDist.getCount("item1"), 1e-2);
    Assert.assertEquals(4, testDist.getTotalCount());
    Assert.assertEquals(Math.log(10d), testDist.get("item1"),
        1e-2);
    Assert.assertEquals(Math.log(100d), testDist.getTotal(),
        1e-2);
    Assert.assertEquals(-Math.log(10d),
        testDist.getLogFraction("item1"), 1e-2);
    Assert.assertEquals(1d / 10d, testDist.getFraction("item1"),
        1e-2);
    Assert.assertEquals(testDist.getFraction("item1"),
        Math.exp(testDist.getLogFraction("item1")), 1e-2);

    /*
     * 'set' should simply reset this value
     */
    testDist.set("item1", Math.log(20d), 1);
    testDist.set("item2", Math.log(10d), 1);

    Assert.assertEquals(4, testDist.getTotalCount());
    Assert.assertEquals(Math.log(100d), testDist.getTotal(),
        1e-2);
    Assert.assertEquals(Math.log(2d) - Math.log(10d),
        testDist.getLogFraction("item1"), 1e-2);
    Assert.assertEquals(2d / 10d, testDist.getFraction("item1"),
        1e-2);
    Assert.assertEquals(testDist.getFraction("item1"),
        Math.exp(testDist.getLogFraction("item1")), 1e-2);

    testDist.set("item1", Math.log(10), 2);
    testDist.set("item2", Math.log(20), 4);
    Assert.assertEquals(8, testDist.getTotalCount());
    Assert.assertEquals(Math.log(100d), testDist.getTotal(),
        1e-2);
    Assert.assertEquals(4, testDist.getDomainSize());

    final CountedDataDistribution<String> testDistClone =
        testDist.clone();

    Assert.assertEquals(testDist.getDomainSize(),
        testDistClone.getDomainSize());
    Assert.assertEquals(testDist.getTotal(),
        testDistClone.getTotal());
    Assert.assertEquals(testDist.getTotalCount(),
        testDistClone.getTotalCount());
    Assert.assertEquals(testDist.getLogFraction("item1"),
        testDistClone.getLogFraction("item1"));
    Assert.assertEquals(testDist.getLogFraction("item2"),
        testDistClone.getLogFraction("item2"));
    Assert.assertEquals(testDist.getLogFraction("item3"),
        testDistClone.getLogFraction("item3"));
    Assert.assertEquals(testDist.getLogFraction("item4"),
        testDistClone.getLogFraction("item4"));
    Assert.assertEquals(testDist.getCount("item1"),
        testDistClone.getCount("item1"));
    Assert.assertEquals(testDist.getCount("item2"),
        testDistClone.getCount("item2"));
    Assert.assertEquals(testDist.getCount("item3"),
        testDistClone.getCount("item3"));
    Assert.assertEquals(testDist.getCount("item4"),
        testDistClone.getCount("item4"));

    testDist.increment("item1", Math.log(10));
    Assert.assertEquals(9, testDist.getTotalCount());
    Assert.assertEquals(Math.log(110d), testDist.getTotal(),
        1e-2);

    testDist.increment("item1", Math.log(10), 2);
    Assert.assertEquals(11, testDist.getTotalCount());
    Assert.assertEquals(Math.log(120d), testDist.getTotal(),
        1e-2);

    /*
     * Can't test negative increments; doesn't make sense here.
     * 'decrement' should be used instead.
     */
    //    testDist.increment("item1", -Math.log(10));
    //    AssertJUnit.assertEquals(8, testDist.getTotalCount());
    //    AssertJUnit.assertEquals(Math.log(100d), testDist.getTotal(), 1e-2);

    final Random rng = new Random(1234533l);
    final int numSamples = (int) 1e5;
    final CountedDataDistribution<String> sampleDist =
        new CountedDataDistribution<String>(true);
    for (final String val : testDist.sample(rng, numSamples)) {
      sampleDist.increment(val);
    }

    Assert.assertEquals(numSamples, sampleDist.getTotalCount());
    Assert.assertEquals(testDist.getLogFraction("item1"),
        sampleDist.getLogFraction("item1"), 1e-2);
    //         Math.sqrt(testDist.getFraction("item1") * (1d - testDist.getFraction("item1"))));
    Assert.assertEquals(testDist.getLogFraction("item2"),
        sampleDist.getLogFraction("item2"), 1e-2);
    //         Math.sqrt(testDist.getFraction("item2") * (1d - testDist.getFraction("item2"))));
    Assert.assertEquals(testDist.getLogFraction("item3"),
        sampleDist.getLogFraction("item3"), 1e-2);
    //         Math.sqrt(testDist.getFraction("item3") * (1d - testDist.getFraction("item3"))));
    Assert.assertEquals(testDist.getLogFraction("item4"),
        sampleDist.getLogFraction("item4"), 1e-2);
    //         Math.sqrt(testDist.getFraction("item4") * (1d - testDist.getFraction("item4"))));

    final Random rng2 = new Random(1234533l);
    final CountedDataDistribution<String> sampleDist2 =
        new CountedDataDistribution<String>(true);
    for (int i = 0; i < numSamples; i++) {
      sampleDist2.increment(testDist.sample(rng2));
    }
    Assert.assertEquals(numSamples, sampleDist2.getTotalCount());
    Assert.assertEquals(numSamples,
        Math.exp(sampleDist2.getTotal()), 1e-5);
    Assert.assertEquals(testDist.getLogFraction("item1"),
        sampleDist2.getLogFraction("item1"), 1e-2);
    //         Math.sqrt(testDist.getFraction("item1") * (1d - testDist.getFraction("item1"))));
    Assert.assertEquals(testDist.getLogFraction("item2"),
        sampleDist2.getLogFraction("item2"), 1e-2);
    //         Math.sqrt(testDist.getFraction("item2") * (1d - testDist.getFraction("item2"))));
    Assert.assertEquals(testDist.getLogFraction("item3"),
        sampleDist2.getLogFraction("item3"), 1e-2);
    //         Math.sqrt(testDist.getFraction("item3") * (1d - testDist.getFraction("item3"))));
    Assert.assertEquals(testDist.getLogFraction("item4"),
        sampleDist2.getLogFraction("item4"), 1e-2);
    //         Math.sqrt(testDist.getFraction("item4") * (1d - testDist.getFraction("item4"))));

    testDistClone.clear();
    testDistClone.copyAll(testDist);

    Assert.assertEquals(testDist.getTotalCount(),
        testDistClone.getTotalCount());
  }
}
