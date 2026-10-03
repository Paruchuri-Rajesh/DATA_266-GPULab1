# Task 2 — Manual error review (Rajesh Paruchuri)

**Model reviewed:** Experimental A, BiGRU + attention (`checkpoints/exp_a_bigru_attn_full.pt`).
It was chosen because it had the best validation macro-F1 of my two experimental models (0.9518 vs 0.9321); the choice was made on validation data, not test.
The 20 cases come from `outputs/error_cases_exp_a_bigru_attn_full.json`, picked by the last cells of the notebook:
* 5 most confident false positives (true negative, highest p(pos))
* 5 most confident false negatives (true positive, lowest p(pos))
* 5 errors closest to the 0.5 threshold
* 5 random errors (seed 266) from the model's worst slice. That slice is **non-English** (133 test reviews, error rate 17.3% vs 4.8% overall).

**Labelling convention matters here.** Yelp polarity labels **1–2 stars as negative and 3–4 stars as positive**.
A lukewarm 3-star review is therefore "positive", and a mildly disappointed 2-star review is "negative".

| # | Group | True | p(pos) | Other models p(pos): fastText / Transformer | Error type |
|---|---|---|---|---|---|
| 01 | confident FP | neg | 1.000 | 1.00 / 1.00 | Label noise (text contradicts rating) |
| 02 | confident FP | neg | 1.000 | 0.89 / 0.92 | Positive surface words, negative price/value verdict (sarcasm) |
| 03 | confident FP | neg | 1.000 | 0.93 / 0.99 | Text–rating mismatch: almost all-praise text, low star rating |
| 04 | confident FP | neg | 0.999 | 0.89 / 0.92 | Hedged, mixed review at the 2-vs-3-star boundary |
| 05 | confident FP | neg | 0.999 | 0.88 / 0.91 | Expectation violation: long positive detail, negative verdict at the end |
| 06 | confident FN | pos | 0.000 | 0.03 / 0.04 | Rating rests on one explicit sentence ("keep the four star rating") |
| 07 | confident FN | pos | 0.001 | 0.01 / 0.14 | Implicit positive: hyperbolic complaint about losing a liked item |
| 08 | confident FN | pos | 0.001 | 0.00 / 0.00 | Label noise: edited review, star rating not updated |
| 09 | confident FN | pos | 0.001 | 0.01 / 0.01 | Mixed/constructive 3-star review dominated by complaints |
| 10 | confident FN | pos | 0.001 | 0.03 / 0.01 | Sarcasm/irony; signal lost by lowercasing and punctuation removal |
| 11 | near threshold | pos | 0.500 | 0.33 / 0.67 | Balanced mixed review (3-star "okay") |
| 12 | near threshold | neg | 0.501 | 0.39 / 0.41 | Mild/hedged negative; minimiser "just OK" lost to stopwording |
| 13 | near threshold | neg | 0.501 | 0.96 / 0.99 | Sarcasm + price complaint (digits removed) |
| 14 | near threshold | neg | 0.502 | 0.41 / 0.51 | Non-English (French), mild negative |
| 15 | near threshold | neg | 0.502 | 0.21 / 0.92 | Mixed; explicit "Not recommended" and grade "C-" under-weighted |
| 16 | slice: non-English | neg | 0.998 | 0.44 / 0.74 | Language shift (French): English stemming/tokenising destroys words |
| 17 | slice: non-English | pos | 0.128 | 0.54 / 0.71 | Language shift: French praise unread; French negation read as English |
| 18 | slice: non-English | pos | 0.307 | 0.81 / 0.26 | Language shift: mild French positive |
| 19 | slice: non-English | neg | 0.645 | 0.51 / 0.02 | Language shift: accent splitting breaks "Exécrable" |
| 20 | slice: non-English | neg | 0.865 | 0.43 / 0.58 | Language shift + mixed ("délicieux" but tiny portions) |

---

## A. Confident false positives (true negative, model sure it is positive)

**01. p = 1.000, 23 words.**
> "Wow love the place and everything is very clean and new! Great place to come and relax worth a try! Cheers, Eric …"

- **Type:** label noise. The text contains no negative content at all, and all three of my models give p ≥ 0.999.
  The reviewer most likely misclicked the star rating, or the review was attached to the wrong business.
  No text classifier should "fix" this.
- **Testable fix:** pull every training review on which all three models disagree with the label at p > 0.99.
  Hand-audit 200 of them to estimate the noise rate, then retrain with label smoothing ε = 0.1.
  Expected: Brier and ECE improve on test, and accuracy changes by less than 0.1 pp.

**02. p = 1.000, 191 words.**
> "My rib eye was very tasty… If you were not born with a silver spoon in your mouth… you should probably go somewhere else… Service good, Ambiance great, Food Fantastic, Service A+, Bottom line, Oh Damn! Bring the Platinum or the Black Card."

- **Type:** praise of quality, negative verdict on value, delivered as sarcasm.
  The review's real complaint is the $99 pork chop and $199 steak.
  My preprocessing removes all digits and `$`, so the prices the reviewer is angry about never reach the model.
  What remains is "tasty… fantastic… great… A+".
- **Testable fix:** map currency amounts to a `<price>` token plus a log-price bucket (`<price_100+>`) instead of deleting them.
  Measure macro-F1 on the test slice of reviews containing `$` followed by a number, before and after.

**03. p = 1.000, 193 words.**
> "The bakery here is tops… Huge brownies… The best thing I found is the individual pineapple upside-down cakes… you'll never buy pizzeria pizza ever again… The deli isn't the greatest but it will do in a pinch."

- **Type:** text–rating mismatch. Over 90% of the text is praise; the only criticism is one mild clause.
  Rated 1–2 stars by a reviewer who scores harshly. It is close to label noise, but the review is genuine.
- **Testable fix:** as in 01, a confident-disagreement audit. Or add a reviewer-severity feature (that reviewer's mean star rating) if user IDs were available.
  On this text-only dataset, the realistic test is the audit: what share of the p > 0.99 disagreements are text–rating mismatches rather than model errors.

**04. p = 0.999, 176 words.**
> "you can't beat it for the price and location. But yes, you get what you pay for!… drains in the shower and the sink backed up… Would I stay here again, probably. ;)"

- **Type:** hedged, mixed review at the 2-vs-3-star boundary. The tone is good-natured, and "probably" stay again reads as positive.
  It is a 2-star by the user's scale. The attention pooling picked up "can't beat it", "perfect place" and "nicer".
- **Testable fix:** train on the 5-class Yelp star labels with an ordinal loss (cumulative-link or CORAL), then collapse to binary.
  The model then learns where 2 and 3 stars differ, instead of treating "2" and "1" as identical.
  Measure error rate on test reviews with p(pos) between 0.2 and 0.8, the current confusion band.

**05. p = 0.999, 422 words.**
> "…Those empanadas are probably the best part of the menu! … After that I was a little under-enthused… You would think a Top Chef Masters restaurant would be more impressive… I expected this place to knock my socks off!… the food has been less than piping hot."

- **Type:** expectation violation. The first half is detailed praise. The verdict is built from counterfactuals ("you would think", "I expected… to knock my socks off") and an understatement ("less than piping hot").
  Each of those phrases contains a positive word ("impressive", "knock my socks off", "hot").
- **Testable fix:** add the expectation and counterfactual cue words ("would think", "expected", "supposed to", "should have") to a contrast-cue slice, and add 1–2 synthetic training examples per pattern.
  Measure the slice error rate before and after.
  An attention check is also cheap: if the attention weight on the final 25% of tokens is low for these errors, a "last-sentence" feature is the fix to test.

## B. Confident false negatives (true positive, model sure it is negative)

**06. p = 0.000, 215 words.**
> "The food is crap… probably the worst nachos I've ever had… Hell, just go on Thursday and drink until it all tastes good… Obviously, Thursday night gets to keep the four star rating. … I will never get too old for Barney's Thursday nights."

- **Type:** the rating is stated explicitly in one sentence and contradicts 90% of the text.
  The reviewer rates the Thursday-night bar, not the food.
  No sentiment model reading word polarity will recover "four star" from the tokens `four`, `star`.
- **Testable fix:** a regex feature for explicit rating mentions ("N star(s)", "N/5", "A+", "one star", …), appended as a token such as `<rating_4>`.
  Test on the slice of test reviews containing such a mention (I can count it with the same regex).

**07. p = 0.001, 22 words.**
> "I am really pissed they don't have the fresh baked pecan chocolate chip cookies anymore...... someone needs to be shot for this."

- **Type:** implicit positive sentiment. Anger about losing an item shows attachment to the place; it is a 3–4 star review.
  Every word with polarity ("pissed", "shot", "don't") is negative.
- **Testable fix:** this "they don't have X anymore" construction is a known blind spot.
  Write a 50-review contrast set ("they stopped selling X, devastated" with positive labels).
  Measure accuracy on it before and after adding 500 matched synthetic examples to training.
  The contrast set also guards against fixing it by memorising one phrase.

**08. p = 0.001, 96 words.**
> "EDIT: They really did change the service up since I last posted this. Horrible service. Used to be my favorite pizza in the city… we're not children."

- **Type:** label noise from an edited review. The text was rewritten to be negative, but the old 3–4 star rating stayed.
  The model is right about the current text.
- **Testable fix:** detect `^EDIT|UPDATE` reviews, measure their error rate (expected well above 4.8%), and drop them from training.
  Then check whether test accuracy on non-edited reviews changes.

**09. p = 0.001, 275 words.**
> "There is no denying that the food is great... but I am back to concerned about the service… the plate looked like a train wreck… Again food is really tasty but the inconsistencies will lose customers. Cary runs her butt off…"

- **Type:** mixed, constructive 3-star review. The volume of complaint text (wait times, dust, empty bar) outweighs two sentences of food praise, and the label boundary sits at 3 stars.
- **Testable fix:** the same ordinal-label training as case 04.
  This is the most common pattern in my errors, so the ordinal model is the single highest-value experiment.

**10. p = 0.001, 88 words.**
> "TERRIBLE SERVICE, RUDE WAITERS WITH A PISS POOR ATTITUDE! WOULD EAT HERE AGAIN! A++++ … What makes this place awesome is the atmosphere… the very obnoxious staff and the hats. Got to love the free hats."

- **Type:** sarcasm/irony. The restaurant's gimmick is rude staff, and the reviewer loves it.
  My preprocessing lowercases everything and removes `+`, `!` and `;)`, so the cues that mark irony are erased ("A++++", the ALL-CAPS mock outrage followed by "WOULD EAT HERE AGAIN!", the wink).
- **Testable fix:** keep emphasis signals as tokens: `<allcaps>` before shouted words, `<grade_a+>` for letter grades, `<wink>` for `;)`, and `<exclaim>` for runs of "!".
  Measure macro-F1 on the slice of test reviews containing an all-caps run of 3 or more words.

## C. Near-threshold errors (|p − 0.5| < 0.002)

**11. p = 0.4998, true pos.**
> "Pho Tai Chin is pretty good. Banh tam bi is so good, but the portion is so small… Price - $7 for pho is higher than usual  Sevice - okay… a clean Vietnamese restaurant that I can count on for flavor."

- **Type:** genuinely balanced 3-star review with item-by-item pros and cons. The model is right to be unsure; the label is "positive" only because 3 stars count as positive.
- **Testable fix:** tune the decision threshold on validation instead of fixing it at 0.5. The 3-star-heavy positive class suggests an optimum slightly below 0.5.
  Report test accuracy at the validation-optimal threshold.

**12. p = 0.5006, true neg.**
> "Sorry to say, that I am not a big fan… chewed on ice crystals… the inconsistency kind of irked me… Overall, the vanilla was just OK."

- **Type:** mild, polite negative.
  The summary "just OK" is the verdict, but **"just" is on the NLTK stopword list and I remove it**, so the model sees "ok", which leans positive.
  "Not a big fan" survives (I keep "not") but is softened by "big fan".
- **Testable fix:** add minimisers ("just", "only", "barely", "hardly") to `KEEP_WORDS`; "only" is already kept.
  Measure on the slice of test reviews containing "just ok", "just okay" or "just fine".

**13. p = 0.5013, true neg.**
> "Great place for folks with fat, fat wallets. They'll gladly relieve the strain! I just wanted a glass of Merlot… "that comes to $19.50. NEVER AGAIN!"

- **Type:** sarcasm plus a price complaint, again with the price removed by preprocessing.
  The fastText baseline (0.96) and Transformer (0.99) are fooled worse; the BiGRU at least lands on the threshold, probably from "never again".
- **Testable fix:** the same `<price>` token fix as case 02, plus keeping `<allcaps>` (case 10).
  These two errors share a root cause and can be tested together on the `$`-amount slice.

**14. p = 0.5019, true neg.** French review of a Mexican restaurant ("j'ai sincèrement trouvé ça moyen…").
- **Type:** non-English input (see section D). Most French words are out-of-vocabulary or mangled by the English stemmer, so p ≈ 0.5 means "no signal".
- **Testable fix:** see section D (Unicode-aware tokeniser).

**15. p = 0.5019, true neg.**
> "Rooms were clean and reasonably priced. Location wise a C-… Very loud at night… Not recommended if you want to get some sleep. Not recommended for families"

- **Type:** mixed review whose verdict ("Not recommended" twice, a C- grade) comes at the end.
  The baseline got this right (0.21), because the bigram "not recommend" is a strong negative feature for it.
  The BiGRU softened it with the positive opening.
- **Testable fix:** average the baseline's and the BiGRU's probabilities.
  Their McNemar discordance is large (745 vs 776 cases where only one of them is right), so they make different mistakes.
  Measure the ensemble's test macro-F1 against the BiGRU's 0.9523.
  This is cheap and directly testable from the saved prediction files.
  **Post-hoc check:** the plain average reaches **0.9572** test macro-F1 (+0.5 pp over the BiGRU). It still needs confirming on validation before it is adopted.

## D. Slice-specific failures — non-English reviews (worst slice)

On the 133 non-English test reviews, BiGRU macro-F1 is **0.810** (error 17.3%), against 0.952 overall.
The baseline does better here (0.875), and the Transformer does worse (0.781).

**16. p = 0.998, true neg.**
> "Hum, cette attraction a peut etre connu jadis son heure de gloire, mais aujourd'hui, ce n'est plus vraiment ca… Les chorégraphies sont basiques… assez vulgaire… les effets spéciaux sont plutot réussis…"

Confidently wrong. The negative content is French ("basiques", "vulgaire", "ce n'est plus"), and none of it is in an English-built vocabulary.
English-looking tokens that survive ("las vegas", "strip", "special") carry positive priors.

**17. p = 0.128, true pos.**
> "Tri Expresse est LA meilleure place à sushi… Voici les points négatifs… Pas de payement direct… C'est absolument charmant."

The positive French ("meilleure", "excellent", "charmant") is barely seen in training.
But French "pas" (not) and "négatifs" are close to English-looking negative tokens, so the model reads the con list and misses the praise.

**18. p = 0.307, true pos.** "J'aime bien le Palais des congrès… il manque de toilettes…"
A mild French positive; "j'aime bien" is unknown to the model, while "manque" (lack) and the toilet complaint dominate.

**19. p = 0.645, true neg.** "…pâtes trop cuites avec de l'eau au fond de l'assiette… Exécrable!"
My cleaning step keeps only `a–z`, so "**Exécrable**" (the review's verdict) becomes the two tokens `ex`, `crabl`, and "pâtes" becomes `p`, `tes`.
The accent-stripping regex destroys exactly the most informative word.

**20. p = 0.865, true neg.** "Les portions sont lilliputiennes… je suis restée sur ma faim… C'était par contre délicieux… J'ai aimé le décor"
A mixed French review. "délicieux" is close to the frequent English stem "delici" once its accent is split off, so the one positive clause dominates.

**Error type (16–20):** domain/language shift.
The training distribution is over 99% English. My English stopword list, English Snowball stemmer and `[^a-z]` filter actively damage French text.

**One testable fix for the slice:** replace the `[^a-z]` filter with a Unicode-letter filter (`[^\w]` with `re.UNICODE`, so "exécrable" stays one token), and apply the French Snowball stemmer to reviews flagged by `is_non_english()`.
The success metric is macro-F1 on the `non_english` test slice (currently 0.810). Overall test macro-F1 must not drop.
The training file has 1,910 non-English reviews (0.34%), enough to learn from without translation.

---

## Summary of error types (20 cases)
| Error type | Cases | Fixable by the model? |
|---|---|---|
| Mixed / hedged sentiment at the 2-vs-3-star boundary | 04, 09, 11, 12, 15 | partly: ordinal labels, threshold tuning |
| Non-English input | 14, 16, 17, 18, 19, 20 | yes: Unicode tokeniser + per-language stemming |
| Sarcasm / irony / implicit sentiment | 02, 07, 10, 13 | partly: keep caps, prices and grades as tokens |
| Label noise / text–rating mismatch / stale edit | 01, 03, 08 | no: audit labels, label smoothing |
| Explicit rating or expectation-violation discourse | 05, 06 | yes: rating-mention feature, cue slice |

The biggest single lesson: **6 of my 20 errors were caused by my own preprocessing.**
- Digits and `$` were removed (cases 02, 13).
- Case and emphasis were removed (case 10).
- "just" was stopworded (case 12).
- Accents were split (cases 19, 20).

These preprocessing choices help on average English text (OOV rate 0.34%) but cost exactly the reviews whose meaning sits in those characters.
