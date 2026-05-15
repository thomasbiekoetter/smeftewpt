(* Transform expression to list with each term in the expression *)
ToTerms[expr_] := If[Head[expr] === Plus, List @@ expr, {expr}];

(* Carry out SumOver operation on each term separately *)
EvSumOverTerm[x_] := Module[
   {instancesSumOver, y, a, b},

   (* Replace colour index summations *)
   (* Assuming that it is always just a factor of 3 *)
   y = x /. SumOver[SUNFIndex[a_], 3] :> 3;

   (* Collect flavour indices to be summed over *)
   instancesSumOver = Cases[y, SumOver[__], Infinity];
   numinds = Length[instancesSumOver];

   (* Remove SumOvers *)
   y = y /. SumOver[___] -> 1;

   (* Rename indices *)
   inds = {};
   Do[
    a = instancesSumOver[[i, 1]];
    b = ToExpression["i" <> ToString[a[[2]]]];
    y = y /. a -> b;
    AppendTo[inds, b];
    , {i, 1, numinds}];

   (* Add proper sums *)
   Do[
    y = Summ[y, {inds[[i]], 1, 3}];
    , {i, 1, Length[instancesSumOver]}];

   (* Carry out summation *)
   y = y /. Summ -> Sum
   ];

(* Sum the terms to get result for full expression *)
EvSumOver[x_] := Total[EvSumOverTerm /@ ToTerms[x]];
