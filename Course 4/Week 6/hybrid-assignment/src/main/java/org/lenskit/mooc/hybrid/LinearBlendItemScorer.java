package org.lenskit.mooc.hybrid;

import com.google.common.base.Preconditions;
import org.lenskit.api.ItemScorer;
import org.lenskit.api.Result;
import org.lenskit.api.ResultMap;
import org.lenskit.basic.AbstractItemScorer;
import org.lenskit.bias.BiasModel;
import org.lenskit.results.Results;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Item scorer that computes a linear blend of two scorers' scores.
 *
 * <p>This scorer takes two underlying scorers and blends their scores.
 */
public class LinearBlendItemScorer extends AbstractItemScorer {
    private final BiasModel biasModel;
    private final ItemScorer leftScorer, rightScorer;
    private final double blendWeight;

    /**
     * Construct a popularity-blending item scorer.
     *
     * @param bias   The baseline bias model to use.
     * @param left   The first item scorer to use.
     * @param right  The second item scorer to use.
     * @param weight The weight to give popularity when ranking.
     */
    @Inject
    public LinearBlendItemScorer(BiasModel bias,
                                 @Left ItemScorer left,
                                 @Right ItemScorer right,
                                 @BlendWeight double weight) {
        Preconditions.checkArgument(weight >= 0 && weight <= 1, "weight out of range");
        biasModel = bias;
        leftScorer = left;
        rightScorer = right;
        blendWeight = weight;
    }

    @Nonnull
    @Override
    public ResultMap scoreWithDetails(long user, @Nonnull Collection<Long> items) {
        List<Result> results = new ArrayList<>();

        for (long i : items) {
            double b_ui = biasModel.getIntercept() + biasModel.getItemBias(i) + biasModel.getUserBias(user);
            double leftScore = 0;
            if (leftScorer.score(user, i) != null) leftScore = leftScorer.score(user, i).getScore() - b_ui;
            double rightScore = 0;
            if (rightScorer.score(user, i) != null) rightScore = rightScorer.score(user, i).getScore() - b_ui;
            double s_ui = b_ui + (1.0 - blendWeight) * leftScore + blendWeight * rightScore;
            results.add(Results.create(i, s_ui));
        }

        return Results.newResultMap(results);
    }
}
