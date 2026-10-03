# Task 1 — Failure analysis (Rajesh Paruchuri)

Model: 6-layer / 6-head / 192-dim character GPT, best checkpoint `checkpoints/best.pt` (epoch 10, val CE 0.711).
All snippets are copied unedited from `outputs/generated_samples_full.txt`.
That file holds 5 prompts × 3 decoders (greedy, T = 0.7, T = 1.0), up to 400 new characters each.

The output is mostly fluent: spelling, punctuation, dialogue quotes and the TinyStories story template are almost always right.
The failures below sit at the **phrase and discourse level**, not the character level.

---

## Case 1 — Degenerate repetition loop (greedy decoding)

**Sample 07, greedy, prompt "Tom and his dog went to the park."**
```
"OK, but be careful. The tree is not a tree. It is a tree. It is a tree. It is a big tree.
It is a big tree. It is a big tree. It is a big tree. It is a big tree. It is a big tree. ...
```
Samples 01, 04 and 13 show the same pattern ("It is a big box. It is so big and shiny." ×4, "You have to be careful and not to be careful." ×4).

**Failure type:** repetition / mode collapse.

**Observation.** Four of the five greedy samples (01, 04, 07, 13) fall into a loop and stay in it until the 400-character limit. The fifth (sample 10) repeats a whole sentence once, then ends at `<|eos|>`.
The repeated word-4-gram rate is **0.396 for greedy versus 0.005 at T = 0.7** (`metrics_report.csv`), and greedy distinct-2 is only 0.38.
Once "It is a big tree." has been produced, the most likely next characters are the ones that start the same sentence again.
Each repetition makes the copy even more likely: the attention heads can find and copy the previous occurrence inside the 192-char window, which is a self-reinforcing feedback loop.
Greedy decoding has no way out because it never takes a lower-probability branch.
The loss does not penalise this: per-character CE stays low even when the text is useless at the sentence level.

**Testable fix.** Decode with T = 0.7 (already 80× fewer repeated 4-grams here), or add a repetition penalty / no-repeat-4-gram constraint to greedy.
Success criterion: greedy repeated-4-gram rate drops below 0.05 with no rise in val CE.

---

## Case 2 — Contradiction and lost speaker/entity tracking

**Sample 13, greedy, prompt `Mom said, "`**
```
Mom said, "Let's go to the park with the ball. It will be fun to play with."
Mom said, "Okay, Mom. I will be friends. But you have to be careful with the ball.
You have to be careful and not to be careful.
```
**Sample 02, T = 0.7:** the story opens with "a little boy named Timmy" and three sentences later continues "*Lily* looked at the tree and said, …". Lily was never introduced.
**Sample 11, T = 0.7:** "The sun smiled and said, "I'm Lily, I need to do something special.""

**Failure type:** loss of coherence (contradiction, speaker confusion, entity drift).

**Observation.**
- "Mom" answers herself and calls the listener "Mom".
- A sentence asserts and negates the same thing: "be careful and not to be careful", and in sample 07 "The tree is not a tree. It is a tree."
- Main characters swap mid-story.

Each clause is grammatical and typical of TinyStories, but the model does not keep a consistent picture of who is speaking or what is true.
A character model with 2.7M parameters spends most of its capacity on spelling and local syntax.
"Lily" is simply the most frequent name in the corpus, so it is a high-prior completion after `said,`.
The 192-character context also means the opening of a 400-character sample has left the window by the end.

**Testable fix.** Give the model more context and capacity per word:
- either a longer block (384) or a subword/BPE tokenizer (the same 192 positions would then cover about 4× more text);
- then measure the share of sampled stories that switch to a protagonist the prompt or first sentence never introduced (currently 3 of 10: samples 02 Timmy→Lily, 05 Lily→Anna, 08 Tom→Lily).

---

## Case 3 — Invented words and semantically impossible events (T = 1.0)

**Sample 09, T = 1.0:** "They saw a big man who was stuck in the sky. Tom saw a **bloll** on the floor."
**Sample 12, T = 1.0:** "It was dark and spaceship. … That's a small **conter** and food. … Mum saw a guard **car-conor** on the ground. She saw how much the dust started to talk to the umbrella about him."
**Sample 03, T = 1.0:** "her mom said it was too **tisty**"
**Sample 14, T = 0.7:** "they do not care about **wandersting**"

**Failure type:** broken words (character-level hallucination) plus semantic hallucination / broken grammar.

**Observation.** A character-level model spells every word one character at a time, so at T = 1.0 the low-probability tail sometimes wins mid-word.
That produces plausible-looking non-words: "bloll", "conter" (counter/container), "tisty" (tasty), "wandersting" (wandering + interesting).
The same flattening shows at the sentence level: "dust started to talk to the umbrella", "a man stuck in the sky", "It was dark and spaceship" (adjective slot filled by a noun).
Higher temperature buys diversity (distinct-3 = 0.97 at T = 1.0 vs 0.94 at T = 0.7) at the cost of these errors.
T = 0.7 produced 1 non-word across its 5 samples; T = 1.0 produced 4 (tisty, bloll, conter, car-conor).

**Testable fix.** Use nucleus (top-p = 0.9) or top-k = 10 sampling at T = 1.0.
This cuts the low-probability tail without collapsing into Case 1.
Measure the out-of-dictionary word rate (words not in the training vocabulary) and keep distinct-2 above 0.8.

---

### What did *not* fail
- No story-boundary leakage: the explicit `<|eos|>` token works. Samples 10 and 11 end cleanly at `<|eos|>` instead of running on into an unrelated story.
- No `<unk>` characters were generated.
- No NaN/Inf steps, and no loss spikes after warm-up.
