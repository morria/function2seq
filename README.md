# Function2Seq

This codebase is a simplified implementation of [code2seq](https://code2seq.org) and
is based on the tensorflow2 fork [kolkir/code2seq](https://github.com/Kolkir/code2seq).

## Running it

1. First, produce an input file which maps function name tokens to list of "context
   paths" as described by [the paper](https://openreview.net/pdf?id=H1gKYo09tX) such
   as by using [morria/code2seq-paths-php](https://github.com/morria/code2seq-paths-php)
   or [ASTMiner](https://github.com/JetBrains-Research/astminer). The input file should
   have the format described in the section [Input file format].

2. Split the data into training, evaluation and testing datasets via the following.

```
python -m function2seq.split --input-file <input_file.c2s> --output-directory data/input --seed 42 --test-ratio 0.1 --evaluation-ratio 0.1 --max-contexts 200
```

3. Train the model such as via the following.

```
python -m function2seq.train --input-train=data/input/train.c2s --input-validation=data/input/eval.c2s -o data/model/ --seed 42
```

4. Make predictions via the following.

```
python -m function2seq.predict --model data/model # ...
```

## Input file format

The input file is structured as follows.

```
<line> ::= <function_name> WS <context_list>

<function_name> ::= <subtoken_list>

<context_list> ::= <source_terminal> ',' <node_list> ',' <target_terminal>

<source_terminal> ::= <subtoken_list>

<target_terminal> ::= <subtoken_list>

<subtoken_list> ::= string | string '|' <subtoken_list>

<node_list> ::= string | string '|' <node_list>

WS ::= ' '
```

For example, consider the following code.

```php
<?php
function f(int $x)
{
    return $x;
}
```

This code should produce the following input file;

```
f long,parameter,x long,parameter|function|return|variable,x x,parameter,long x,parameter|function|return|variable,x x,variable|return|function|parameter,long x,variable|return|function|parameter,x
```

