package org.lenskit.mooc.cbf;

import org.lenskit.data.ratings.Rating;
import org.lenskit.data.ratings.Ratings;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Build a user profile from all positive ratings.
 */
public class WeightedUserProfileBuilder implements UserProfileBuilder {
    /**
     * The tag model, to get item tag vectors.
     */
    private final TFIDFModel model;

    @Inject
    public WeightedUserProfileBuilder(TFIDFModel m) {
        model = m;
    }

    @Override
    public Map<String, Double> makeUserProfile(@Nonnull List<Rating> ratings) {
        // Create a new vector over tags to accumulate the user profile
        Map<String, Double> profile = new HashMap<>();
        double sumRatings = 0;
        int count = 0;
        for (Rating rating : ratings) {
            sumRatings += rating.getValue();
            count++;
        }
        double meanRating = sumRatings / count;
        for (Rating rating : ratings) {
            Map<String, Double> itemId = model.getItemVector(rating.getItemId());
            for (Map.Entry<String, Double> entry : itemId.entrySet()) {
                String tag = entry.getKey();
                Double normalizedValue = entry.getValue() * (rating.getValue() - meanRating);
                profile.put(tag, profile.containsKey(tag) ? profile.get(tag) + normalizedValue : normalizedValue);
            }
        }
        return profile;
    }
}