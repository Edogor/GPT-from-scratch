# GPT from scratch
language modeling with ngram up to simple GPT model (decoder only transformer) as described in the [Attention is all you need](https://arxiv.org/abs/1706.03762) paper. <br>

In [notebooks/](./notebooks/) you can find a guided runthrough of training and using the models, as well as analyses of the hyperparameter search. <br>

Tokenizer: <br>
- byte level bpe tokenizer [ [from scratch](./bpe_tokenizer.py) | [hf wraper](./bpe_hf.py) ] <br>
  
Models implemented:
- [Ngram model](./ngram_engine) <br>
- [Neural bigram model](./neural_bigram.py) <br>
- [GPT model (decoder only)](./GPT_mj.py) <br>

Data set: [shakespeare](./data/) <br>
- to download and preprocess the data run [clean_nltk_shakespear_data_w_nl.py](./data/clean_nltk_shakespear_data_w_nl.py) from the root (repo) directory: <br> 
  ```python clean_nltk_shakespear_data_w_nl.py``` <br>


run main to train a new <[gpt_model](./gpt_model.py).GPTModel>, by chainging the variables you can also continue the training of a pericular model. <br>
run compare models to see and compare all the models. <br>
run model_ui_gradio for visual interface (has bugs to be fixed) <br>


TODO:
 - put results here and briefly describe them. 
... 
example: <br>
Tiny-GPT training curves: <br>
<img src="./results/GPT/GPT_training_plot.png" alt="GPT training curves" width="720"> <br>

# MilestonesResults

## Unix
    tr 'A-Z' 'a-z' < Shakespeare_clean_full.txt | tr -sc 'A-Za-z' '\n' | sort | uniq -c | sort -n -r

## tokenizer
<img src="./results/tokenizer/tokenizer_search_results.png" alt="GPT training curves" width="720"> <br>
### Explanations (for each subplot)
#### Heaps K vs. merges
- Shows the Heaps’ law constant  𝐾, which reflects vocabulary growth.
- Higher 𝐾 at low merges means many unique tokens; it drops as merges increase.

#### Heaps β vs. merges
- Displays the Heaps’ law exponent β, measuring how fast vocab grows with corpus size.
- β rises with merges, stabilizing at higher values.

#### Utilization vs. merges

- Fraction of vocab actually used.

- Peaks at moderate merges, then declines as vocab becomes too large and sparse.

#### Chars per token vs. merges
- Average number of characters per token.

- Increases with merges (tokens become longer), flattening at ~3 chars/token.

#### Compression ratio vs. merges
- Ratio of model text size to raw text.
- Improves (higher compression) with more merges, but saturates after ~8–10k merges.

#### Tokens per 1k chars vs. merges
- Average number of tokens needed per 1k characters.
- Decreases with merges, showing more compact representation.

### General takeaway

- Low k (few merges): small vocab, many tokens, high utilization.

- High k (many merges): large vocab, fewer tokens, but less efficient utilization.

- Optimal range (~800–1200 merges, green lines): balances efficiency and utilization.

</br>

## N-Gram model
### Hyperparameter search analysis
<img src="./results\ngram\hparams_search\ngram_hparams_fit_results.png" alt="GPT training curves" width="720"> <br>

### Explanation of Results
#### Interpolation Weights (top-left):
- The heatmap shows the learned λ-weights for combining different N-grams. Most weight is concentrated on lower N when merges are small, while higher N contribute more as merges increase.

#### Validation PPL & BPT (top-middle & top-right):
- Both perplexity (PPL) and bits-per-token (BPT) increase steadily as the number of merges grows, meaning larger vocabularies make the N-gram model less predictive and less efficient. The best results are achieved with relatively small merges and moderate N (≈4–5), where PPL and BPT reach their lowest values.

#### Bottom row (Chars per Token, Validation PPL, Validation BPT vs. Merges):
- All three curves confirm the same trend: merges produce longer tokens, but at the cost of higher perplexity and BPT. This shows that for N-gram models, smaller vocabularies provide better predictive power and efficiency, while large vocabularies lead to diminishing returns.

#### Takeaway

- Small to medium merges (≈50–200) with N=4–5 yield the best trade-off between perplexity and efficiency.

- Too few merges → overly fragmented tokens; too many merges → inefficient and poor generalization.

- The optimal tokenizer size for N-gram models is much smaller than for modern neural LMs.

### Test results
<img src=".\results\ngram\hparams_search\ngram_more_test_results.jpg" alt="GPT training curves" width="720"> <br>
#### Perplexity across merges:
- Perplexity increases sharply with the number of merges for train, validation, and test sets alike. The best performance (lowest PPL ≈ 7–8) is achieved with very few merges (k=10). At higher merges (k > 800), perplexity rises strongly, reaching >60 at k=800, indicating that larger vocabularies reduce the predictive quality of the N-gram model.

### Generated Text
#### Overview:
- Few merges (k=10): Very short tokens → fragmented nonsense, many invented words.
- Moderate merges (k=200): More real words, partial structure, but grammar unstable.
- Large merges (k=800): Fluent words and names, but coherence breaks down quickly.

#### Generated Text Examples (N=9, different merge sizes)
Examples (given context: “shall the lovers”):
- merges=10, n=9, lambdas ≈ [~0, ~0, 0.395, 0.605, ~0, ~0, ~0, ~0, ~0]:<br> (fragmented nonsense)

      ?
      Antony with tim.
      Marchoice,
      Alare I'll drance,
      Andeed prime


      Meeds affore me, I drophese?
      An now, with 

  </br>
- merges=200, n=9, lambdas ≈ [~0, 0.541, 0.459, ~0, ~0, ~0, ~0, ~0, ~0]:

      Sce Yound,
      And their cou know, I cangain, I dice,
      BERCAPULET
      MItulavene of my chan that the from the take theyes; and eart.




      Exe to be satter;
      Faphount all my let,
      But to be a riev
</br>

- merges=800, n=9, lambdas ≈ [~0, 1.000, ~0, ~0, ~0, ~0, ~0, ~0, ~0]:

      bese Wheo be senter. If welieceiveder


      And in that we may be,

      Servant


      Ifabinion,






      Thange thee?
      BRUTUS
      H
      Thisonst your dutio is a turn.
      SCOrongue on your corattle;



      What you, and
      C

### neural bigram model

### GPT model


<br><br><br>
# TODO
- pseudocode
