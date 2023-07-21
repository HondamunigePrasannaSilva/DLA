# Laboratory on Large Language Models

This laboratory is about working with Large Language Models (e.g. GPT and BERT) to do various useful things. 

## Warmup Exercise
This code can be found on the notebook: **Lab2-LLMs.ipynb**

## Exercise 1
This first exercise use a *small* autoregressive GPT model for character generation (the one used by Karpathy in his video) to generate text in the style of Dante Aligheri. Using this [file](https://archive.org/stream/ladivinacommedia00997gut/1ddcd09.txt), which contains the entire text of Dante's Inferno.

A generated sample :
```
'al con ostre tanto ben quelle furo
  ch'a ben la rimana appura; bandommi queschi

m'io dise: <Perch'io parla tui non scponda
  ch'i' potrei qui tra doppila quella,
  o susciaremo di che 'n la dotta,
  ch'elli hiserta fe' l'a natto casca>>.

Sa gia` per matta da partien del Bacio
  canta in su lo scoperto, ancor tu mi piace
  perche' sospensava tanto molto,

per consie in cie; e giustizzar li arsi
  se 'l falcun la tua via officio;
  quel l'angoscia no che tanto conento,

per non quel che posa 
```

## Exercise 2
This exercise use the [Hugging Face](https://huggingface.co/) model and dataset ecosystem. 

### Exercise 2.1: Installation and text tokenization

Use `GPT2Tokenizer` to encode the input sentence.
```
sequence = "I enjoy DLA lectures but"
Token length:6
Sequence length: 24
```

### Exercise 2.2: Generating Text

Use pre-trained `GPT2LMHeadModel` to use the [`generate()`](https://huggingface.co/docs/transformers/v4.27.2/en/main_classes/text_generation#transformers.GenerationMixin.generate) method to generate text from a prompt.

```
prompt = "I enjoy DLA lectures but"
```
Using the generate function without beam. 
```
1 - I enjoy DLA lectures but I don't like to spend time with my friends. I'm not a big fan of the "I'm not a big fan of the "I'm not a big fan of the "I'm not a big fan

# with temperture = 0.1
2 - I enjoy DLA lectures but I don't like to spend time with people who are not my friends. I don't like to spend time with people who are not my friends. I don't like to spend time with people who are not my friends

with temperture = 0.9
3 - I enjoy DLA lectures but don't really study the book by itself. Also, I read a lot of books that were on'mystical' topics, and that's not great for my academic career, but I'm not really into esoteric
```
Using the generate function with beam. 

```
4 - I enjoy DLA lectures but I don't think I've ever been able to get to the point where I feel like I'm in the right place."

with temperture = 0.9
5 - I enjoy DLA lectures but I don't think I've ever been in a class like this before."
```

## Exercise 3
This code can be found **textClassification.py**

Finetuning GPT2 model for text classification on **rotten tomatoes** dataset using a linear classifier and a non linear classifier. Model and dataset from hugging face.


to run the code, for more configuration see inside the file. Set linear=True for linear classification otherwise False for **non** linear classifier.

```
$python textClassifcation.py --epochs=20 --linear=True --batch_size=128, --log='disabled'
```

Set log = 'run' if you want to log using wandb


### Plot using linear vs non linear layer on top of GPT2 model
