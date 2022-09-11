package org.lenskit.mooc.hybrid;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.lenskit.api.ItemScorer;
import org.lenskit.api.Result;
import org.lenskit.bias.BiasModel;
import org.lenskit.bias.UserBiasModel;
import org.lenskit.data.ratings.Rating;
import org.lenskit.data.ratings.RatingSummary;
import org.lenskit.inject.Transient;
import org.lenskit.util.ProgressLogger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.inject.Inject;
import javax.inject.Provider;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Trainer that builds logistic models.
 */
public class LogisticModelProvider implements Provider<LogisticModel> {
    private static final Logger logger = LoggerFactory.getLogger(LogisticModelProvider.class);
    private static final double LEARNING_RATE = 0.00005;
    private static final int ITERATION_COUNT = 100;

    private final LogisticTrainingSplit dataSplit;
    private final BiasModel baseline;
    private final RecommenderList recommenders;
    private final RatingSummary ratingSummary;
    private final int parameterCount;
    private final Random random;

    @Inject
    public LogisticModelProvider(@Transient LogisticTrainingSplit split,
                                 @Transient UserBiasModel bias,
                                 @Transient RecommenderList recs,
                                 @Transient RatingSummary rs,
                                 @Transient Random rng) {
        dataSplit = split;
        baseline = bias;
        recommenders = recs;
        ratingSummary = rs;
        parameterCount = 1 + recommenders.getRecommenderCount() + 1;
        random = rng;
    }

    @Override
    public LogisticModel get() {
        List<ItemScorer> scorers = recommenders.getItemScorers();
        double intercept = 0;
        double[] params = new double[parameterCount];
        LogisticModel current = LogisticModel.create(intercept, params);

        List<Rating> tuneRatings = dataSplit.getTuneRatings();

        ArrayList<RealVector> x_array = new ArrayList<>();
        for (int j = 0; j < tuneRatings.size(); j++) {
            Rating r = tuneRatings.get(j);
            long itemId = r.getItemId();
            long userId = r.getUserId();
            double b_ui = baseline.getIntercept() + baseline.getItemBias(itemId) + baseline.getUserBias(userId);
            double lg_popularity = Math.log10(ratingSummary.getItemRatingCount(itemId));

            RealVector x = new ArrayRealVector(parameterCount);
            x.setEntry(0, b_ui);
            x.setEntry(1, lg_popularity);
            int i = 2;
            for (ItemScorer scorer : scorers) {
                Result score_result = scorer.score(userId, itemId);
                if (score_result == null) {
                    x.setEntry(i, 0.);
                    i += 1;
                    continue;
                }
                double x_value = score_result.getScore() - b_ui;
                x.setEntry(i, x_value);
                i += 1;
            }
            x_array.add(x);
        }

        ArrayList<Integer> indexes = new ArrayList<>();
        for(int i = 0; i < tuneRatings.size(); i++)
            indexes.add(i);

        for (int itera = 0; itera < ITERATION_COUNT; itera++) {
            Collections.shuffle(indexes);
            for (int i : indexes) {
                Rating r = tuneRatings.get(i);
                RealVector x = x_array.get(i);
                double y = r.getValue();
                double sigmoid = current.evaluate(-y, x);

                intercept += LEARNING_RATE * y * sigmoid;
                for (int j = 0; j < parameterCount; j++) {
                    params[j] += LEARNING_RATE * y * x.getEntry(j) * sigmoid;
                }
                current = LogisticModel.create(intercept, params);
            }
        }

        return current;
    }

}
