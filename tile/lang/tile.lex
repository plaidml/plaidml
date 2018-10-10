%{

#include <string>
#include "tile/lang/parser.y.h"

%}

%option reentrant bison-bridge
%option noyywrap never-interactive


id        [A-Z][a-zA-Z0-9_]*'?
iid       [a-z][a-zA-Z0-9_]*'?
sid       #[A-Za-z][a-zA-Z0-9_]*'?
siid      $[A-Za-z][a-zA-Z0-9_]*'?
float     [0-9]+(\.[0-9]+|([\.0-9]+)?[eE][-+]?[0-9]+)
int       [0-9]+
white     [ \t\r]
newline   [\n]

%%
"function"    { return FUNCTION; }
"polynomial"  { return POLYNOMIAL; }
"expression"  { return EXPRESSION; }
"contraction" { return CONTRACTION; }
"default"     { return DEFAULT; }
"no_defract"  { return NO_DEFRACT; }

{float} { yylval->s = yytext; return FLOAT_LITERAL; }
{int}   { yylval->i = std::atol(yytext); return INT_LITERAL; }
{id}    { yylval->s = yytext; return ID; }
{iid}   { yylval->s = yytext; return IDXID; }
{sid}   { yylval->s = yytext + 1; return ID; }
{siid}  { yylval->s = yytext + 1; return IDXID; }
"."     { return DOT; }
"*"     { return MULT; }
"/"     { return DIV; }
"+"     { return PLUS; }
"-"     { return MINUS; }
"=="    { return EQ; }
"!="    { return NE; }
"="     { return ASSIGN; }
">"     { return GT; }
"<"     { return LT; }
">="    { return GE; }
"<="    { return LE; }
"->"    { return ARROW; }
"["     { return LBRACE; }
"]"     { return RBRACE; }
"[["    { return LLBRACE; }
"]]"    { return RRBRACE; }
"("     { return LPAREN; }
")"     { return RPAREN; }
"{"     { return LCURLY; }
"}"     { return RCURLY; }
","     { return COMMA; }
"&"     { return BIT_AND; }
"|"     { return BIT_OR; }
"^"     { return BIT_XOR; }
"~"     { return BIT_NOT; }
"<<"    { return BIT_LEFT; }
">>"    { return BIT_RIGHT; }
":"     { return COLON; }
";"     { return SEMICOLON; }
"?"     { return QUESTION; }
{white} {}
{newline} {}
