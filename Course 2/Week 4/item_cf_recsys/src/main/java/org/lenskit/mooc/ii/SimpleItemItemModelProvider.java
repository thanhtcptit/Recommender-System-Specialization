package org.lenskit.mooc.ii;

import com.google.common.collect.Maps;
import it.unimi.dsi.fastutil.longs.Long2DoubleMap;
import it.unimi.dsi.fastutil.longs.Long2DoubleOpenHashMap;
import org.apache.commons.lang3.tuple.Pair;
import org.lenskit.data.dao.DataAccessObject;
import org.lenskit.data.entities.CommonAttributes;
import org.lenskit.data.ratings.Rating;
import org.lenskit.data.ratings.Ratings;
import org.lenskit.inject.Transient;
import org.lenskit.util.IdBox;
import org.lenskit.util.collections.LongUtils;
import org.lenskit.util.io.ObjectStream;
import org.lenskit.util.math.Vectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.inject.Inject;
import javax.inject.Provider;
import java.util.List;
import java.util.HashSet;
import java.util.Map;

/**
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class SimpleItemItemModelProvider implements Provider<SimpleItemItemModel> {
    private static final Logger logger = LoggerFactory.getLogger(SimpleItemItemModelProvider.class);

    private final DataAccessObject dao;

    /**
     * Construct the model provider.
     *
     * @param dao The data access object.
     */
    @Inject
    public SimpleItemItemModelProvider(@Transient DataAccessObject dao) {
        this.dao = dao;
    }

    /**
     * Construct the item-item model.
     *
     * @return The item-item model.
     */
    @Override
    public SimpleItemItemModel get() {
        Map<Long, Long2DoubleMap> itemVectors = Maps.newHashMap();
        Long2DoubleMap itemMeans = new Long2DoubleOpenHashMap();

        try (ObjectStream<IdBox<List<Rating>>> stream = dao.query(Rating.class)
                .groupBy(CommonAttributes.ITEM_ID)
                .stream()) {
            for (IdBox<List<Rating>> item : stream) {
                long itemId = item.getId();
                List<Rating> itemRatings = item.getValue();
                Long2DoubleOpenHashMap ratings = new Long2DoubleOpenHashMap(Ratings.itemRatingVector(itemRatings));

                double mean = Vectors.mean(ratings);
                itemMeans.put(itemId, mean);

                for (Map.Entry<Long, Double> entry : ratings.entrySet()) {
                    entry.setValue(entry.getValue() - mean);
                }
                itemVectors.put(itemId, LongUtils.frozenMap(ratings));
            }
        }

        Map<Long, Long2DoubleMap> itemSimilarities = Maps.newHashMap();
        for (Map.Entry<Long, Long2DoubleMap> item1 : itemVectors.entrySet()) {

            Long2DoubleMap similarities = new Long2DoubleOpenHashMap();
            for (Map.Entry<Long, Long2DoubleMap> item2 : itemVectors.entrySet()) {
                Double similarity = calculateCosineSimilarity(item1.getValue(), item2.getValue());
                if (similarity > 0) {
                    similarities.put(item2.getKey(), similarity);
                }
            }
            itemSimilarities.put(item1.getKey(), similarities);
        }
        return new SimpleItemItemModel(LongUtils.frozenMap(itemMeans), itemSimilarities);
    }

    private Double calculateCosineSimilarity(Long2DoubleMap ratings1, Long2DoubleMap ratings2) {
        double v1v2 = 0.0;
        double v1v1 = 0.0;
        double v2v2 = 0.0;

        HashSet<Long> commonItems = new HashSet<>();
        commonItems.addAll(ratings1.keySet());
        commonItems.addAll(ratings2.keySet());

        for (long item : commonItems) {
            double u1 = 0.0;
            double u2 = 0.0;
            if (ratings1.containsKey(item)) {
                u1 = ratings1.get(item);
            }
            if (ratings2.containsKey(item)) {
                u2 = ratings2.get(item);
            }
            v1v2 += u1 * u2;
            v1v1 += u1 * u1;
            v2v2 += u2 * u2;
        }
        return v1v2 / Math.sqrt(v1v1) / Math.sqrt(v2v2);
    }
}
