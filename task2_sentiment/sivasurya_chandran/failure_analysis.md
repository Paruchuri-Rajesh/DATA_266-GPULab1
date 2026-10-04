# Task 2 error review — Sivasurya Chandran

These are 20 mistakes made by my best model, Experimental 2 (BiLSTM with attention, 95.8% test accuracy). They were picked by `src/error_analysis.py`, and the full list with probabilities is in `checkpoints/error_review_experimental_2.md`. I read each review and assigned an error type myself.

The 20 errors are made up of:
- the 5 most confident false positives;
- the 5 most confident false negatives;
- the 5 predictions closest to 0.5;
- 5 random errors from the worst slice (reviews containing "but/however/although", 4.65% error rate).

P(pos) is the model's probability that the review is positive.

## Summary of error types and fixes

| Error type | Count | Testable fix | How to measure |
|---|---|---|---|
| Contrast clause: the verdict sits on one side of "but/however" | 8 | Add a contrast-aware feature, e.g. mark tokens after "but/however/although" (`but_` prefix), or up-weight the final clause in attention pooling | Error rate on the `contains_but_contrast` slice (currently 4.65%) |
| Label noise: the star rating disagrees with the text | 4 | Estimate the noise rate by hand-labelling 100 random confident errors; train with confident-learning filtering (drop training examples the model confidently disagrees with) | Accuracy on a hand-checked clean test subset; a count of how many confident errors are noise |
| Past-vs-present or story structure: the final verdict differs from most of the text | 2 | Longer context plus position-aware pooling (append the last 64 tokens again, or add a learned position bias to attention) | Error rate on reviews containing "now/used to/anymore/since" |
| Truncation: the verdict comes after the 256-token cut | 3 | `max_seq_len` 512, or head + tail truncation (first 128 + last 128 tokens) | Error rate on the 2.3% of reviews longer than 256 tokens and on `long_reviews` (4.35%) |
| No sentiment signal: too short or off-topic | 2 | Treat 0.4 < P(pos) < 0.6 as "abstain" for very short reviews; add character n-grams for rare words | Accuracy and ECE on the `short_reviews` slice |
| Concession / negation scope | 1 | Mark negation scope during preprocessing (turn the words after "not/no/never" up to the next punctuation into `not_word` tokens), so "not … 2 star" and "too good to disrespect" stop looking like plain negatives | Error rate on the `contains_negation` slice (4.19%) |

## The 20 errors

| # | Bucket | Test # | True → Pred (P(pos)) | Snippet | Error type | Observation |
|---|---|---|---|---|---|---|
| 1 | confident FP | 29330 | neg → pos (0.9999) | "Wow love the place and everything is very clean and new! Great place to come and relax worth a try!" | label noise | The text is entirely positive; the 1–2 star label contradicts it. The model is arguably right. |
| 2 | confident FP | 15988 | neg → pos (0.9997) | "…I find their meats to be generally good, and reasonably generous… Personally, I'm not a fan of loading a sandwich with fries, but again, …" | contrast clause | A measured, polite tone with mostly positive adjectives. The negative verdict is implicit and only emerges in hedges. |
| 3 | confident FP | 17407 | neg → pos (0.9995) | "This was my second time dining at Company. And unfortunately, it will probably be the last… My friend Katie gave me part of her delicious lobster BLT…" | past-vs-present | The opening states the verdict, but most of the text praises the *first* visit; the negative second visit comes later. |
| 4 | confident FP | 12480 | neg → pos (0.9995) | "my husband had an omelette that was good. i had a blt, a little on the small side for $10, but bacon was great. Our server was awesome!" | label noise | The only complaint is minor; the text reads as positive. The star rating carries information the text doesn't. |
| 5 | confident FP | 163 | neg → pos (0.9992) | "This place is defiantly a historic site!… I love learning about the history… Imitation crab & lots of filling. So I don't need to tell you how nasty that was." | contrast clause | The long positive opening outweighs the single strong negative ("nasty") in attention pooling. |
| 6 | confident FN | 22807 | pos → neg (0.0001) | "EDIT: They really did change the service up since I last posted this. Horrible service…" | label noise | The user edited the review to negative but kept the old positive star rating. The text supports the model. |
| 7 | confident FN | 20061 | pos → neg (0.0002) | "Ever wonder what to do if you have lots of extra garbage or recyclables…" (informational, about a waste facility) | truncation | Informational text with little sentiment up front; any recommendation comes after the cut-off. Words like "garbage", "broken" and "trash" pull it negative. |
| 8 | confident FN | 26683 | pos → neg (0.0003) | "…despite still not digging their ordering process, their food is just too good to disrespect with a 2 star review." | concession / negation scope | The positive verdict is expressed through negated negatives ("too good to disrespect", "not … 2 star"). A bag of stemmed tokens (not, disrespect, 2, star) looks negative. |
| 9 | confident FN | 11401 | pos → neg (0.0009) | "Perhaps my expectations were too high… waited forever… brought us first the wrong food and then cold food… it was actually pretty goos… chocolate martinis that were pretty rockin." | label noise | The text is mostly negative with a mild positive ending; a 3→4-star-style rating mapped to positive. It's borderline even for a human. |
| 10 | confident FN | 30793 | pos → neg (0.0009) | "This place is so much better since they changed owners… it was terrible. We waited forever… Now its much better. The staff are very friendly…" | past-vs-present | Most of the text describes the old, bad experience; the positive present-tense verdict comes at the end. |
| 11 | near threshold | 37541 | neg → pos (0.5003) | "Food was good, not great, and we weren't impressed… We'll definitely opt for other BBQ restaurants before returning here." | contrast clause | "good, not great" plus a hedge. The decisive signal is the final intention sentence. |
| 12 | near threshold | 26481 | pos → neg (0.4986) | "My first experience with STK was at the NYC location… a city that has its fair share of trendy restaurants that lack culinary inspiration…" (256 tokens, truncated) | truncation | A long comparison across three locations; the verdict for this location is beyond the 256-token cut. |
| 13 | near threshold | 4439 | pos → neg (0.4986) | "mark & mercedes are a must listen to in the morning." | no sentiment signal | 5 tokens after cleaning; "must" and "listen" are rare in restaurant reviews, so the model has almost no evidence. P ≈ 0.5 is the right uncertainty. |
| 14 | near threshold | 18716 | neg → pos (0.5015) | "where is this place?" | no sentiment signal | 1 token after stopword removal. There is no sentiment in the text at all; the label exists only because of the stars. |
| 15 | near threshold | 6825 | neg → pos (0.5019) | "A very nice guy came… He was professional and respectful. However he did not find the true problem… the fridge leaked a third time" | contrast clause | Praise for the technician's manner versus failure of the service; the two cancel out. |
| 16 | worst slice | 9162 | neg → pos (0.5425) | "Food is good but the portions are small for what you are paying… the service and the ambience… think again." | contrast clause | Positive nouns (food, service, ambience) versus a value complaint after "but". |
| 17 | worst slice | 25670 | neg → pos (0.8618) | "Hate to be the bad review guy, but here goes… Food and service was very nice… succulent filet mignon… no-hat policy…" | contrast clause | The complaint is about a policy, not the food. Explicit praise for the food dominates. |
| 18 | worst slice | 28735 | pos → neg (0.0719) | "Quite possibly the greasiest pizza I've eaten since junior high cafeteria days. But it was good. If I was hankering for pizza, this would do! Smiley face" | contrast clause | The verdict comes after "But", in a short clause; "greasiest" and "cafeteria" dominate. |
| 19 | worst slice | 19109 | neg → pos (0.8475) | "…this 2 star rating is as high as I can go. The lunch buffet is ample, with many vegetarian options… however, every dish I've tried here is just too spicy." | contrast clause | Long, articulate praise of the chiles; the negative is "too spicy" plus an explicit "2 star". The model doesn't learn that "2 star" is negative strongly enough. |
| 20 | worst slice | 30300 | pos → neg (0.1999) | "Great Bao was closing their brick & motor location… odd location inside a Salon… Food and Hair products just sound so wrong…" (256 tokens, truncated) | truncation / past-vs-present | A narrative about the odd location and closing; the "most heavenly Baos" praise is brief, and the final verdict is past the cut-off. |

## What I took from this

- **About 4 of the 20 errors aren't really model errors.** Yelp polarity labels come from stars (1–2 = negative, 4–5 = positive), and sometimes the text says something different. Examples are reviews edited after the fact, and reviews with only a small complaint. Some of the gap between 96% and 100% is due to the labels, not the model.
- **Contrast is the biggest real weakness.** It covers 8 of the 20 errors, and it's the worst slice overall. The BiLSTM already handles it better than the other two models (4.65% error vs 5.44% for the CNN and 7.51% for the baseline). Even so, attention pooling still lets whichever side of the "but" is longer win.
- **Truncation and story structure explain another 5.** In long reviews, the actual verdict often comes at the very end, after the 256-token cut, or after a long description of an earlier visit.
