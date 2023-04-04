## Input File Format

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

