(***********************************************************)
(* O(g^4) dimensional reduction of the dimension-six SMEFT *)
(***********************************************************)

(* Authors: Mikael Chala, Luis Gil and Maria Cristina Fiore *)
(* Institution: FTAE, Universidad de Granada *)

(* If you use these results in your work, we kindly ask you to cite our original article *)

(* Notes: 
- We use a different name for the Wilson coefficients here, but they can easily be identified with the ones used in the article.
- The different functions (SumOver, Index, ...) are FeynCalc functions, so you can use this package to print a more readable version of the results.
- Four-fermion signs were not computed correctly in FeynCalc until very recently (July 2025), when a bug fix to FCFADiracChainJoin was uploaded to their GitHub page. We have not, however, explicitly cross-checked that this fix gets all signs right, so we advice extra care when using the four-fermion results.
*)


(*** One-loop matching (super-renormalizable operators) ***)

OneLoopMatching = {mH2 ->  1/12*mu^2*(cHD - 2*cHbox)*T^2, 
mB02 -> 0, 
mW02 -> 0, 
KH -> -1/6*(cHbox*T^2) + (cHD*T^2)/12, 
KB0 -> (-2*cHB*T^2)/3, KB -> (-2*cHB*T^2)/3, 
lmbdH4 -> -(cH*T^2) + (cHB*g1^2*T^2)/4 + (cHD*g1^2*T^2)/16 + (cHWB*g1*g2*T^2)/4 + (cHD*g2^2*T^2)/16 + (3*cHW*g2^2*T^2)/4 + cHbox*lmbd*T^2 - (cHD*lmbd*T^2)/3 - 
   (T^2*ceH[Index[Generation, 5], Index[Generation, 6]]*Conjugate[YE[Index[Generation, 5], Index[Generation, 6]]]*SumOver[Index[Generation, 5], 3]*SumOver[Index[Generation, 6], 3])/12 - 
   (T^2*cdH[Index[Generation, 5], Index[Generation, 6]]*Conjugate[YD[Index[Generation, 5], Index[Generation, 6]]]*SumOver[Index[Generation, 5], 3]*SumOver[Index[Generation, 6], 3]*SumOver[SUNFIndex[Col5], 3])/12 - 
   (T^2*Conjugate[YU[Index[Generation, 5], Index[Generation, 6]]]*cuH[Index[Generation, 5], Index[Generation, 6]]*SumOver[Index[Generation, 5], 3]*SumOver[Index[Generation, 6], 3]*SumOver[SUNFIndex[Col5], 3])/12 - 
   (T^2*Conjugate[cdH[Index[Generation, 6], Index[Generation, 5]]]*SumOver[Index[Generation, 5], 3]*SumOver[Index[Generation, 6], 3]*SumOver[SUNFIndex[Col5], 3]*YD[Index[Generation, 6], Index[Generation, 5]])/12 - 
   (T^2*Conjugate[ceH[Index[Generation, 6], Index[Generation, 5]]]*SumOver[Index[Generation, 5], 3]*SumOver[Index[Generation, 6], 3]*YE[Index[Generation, 6], Index[Generation, 5]])/12 - 
   (T^2*Conjugate[cuH[Index[Generation, 6], Index[Generation, 5]]]*SumOver[Index[Generation, 5], 3]*SumOver[Index[Generation, 6], 3]*SumOver[SUNFIndex[Col5], 3]*YU[Index[Generation, 6], Index[Generation, 5]])/12, 
lmbdB04 -> 0, 
lmbdH2B02 -> (cHbox*g1^2*T^2)/8 + (3*cHD*g1^2*T^2)/16 - (g1^2*T^2*cHe[Index[Generation, 5], Index[Generation, 5]]*SumOver[Index[Generation, 5], 3])/6 - (g1^2*T^2*cHl1[Index[Generation, 5], Index[Generation, 5]]*SumOver[Index[Generation, 5], 3])/6 - 
   (g1^2*T^2*cHd[Index[Generation, 5], Index[Generation, 5]]*SumOver[Index[Generation, 5], 3]*SumOver[SUNFIndex[Col5], 3])/18 + (g1^2*T^2*cHq1[Index[Generation, 5], Index[Generation, 5]]*SumOver[Index[Generation, 5], 3]*SumOver[SUNFIndex[Col5], 3])/18 + 
   (g1^2*T^2*cHu[Index[Generation, 5], Index[Generation, 5]]*SumOver[Index[Generation, 5], 3]*SumOver[SUNFIndex[Col5], 3])/9, KW -> (-2*cHW*T^2)/3 - 2*c3W*g2*T^2, KW0 -> (-2*cHW*T^2)/3 - 2*c3W*g2*T^2, 
lmbdH2W02 -> (cHbox*g2^2*T^2)/8 + (cHD*g2^2*T^2)/48 + (g2^2*T^2*cHl3[Index[Generation, 5], Index[Generation, 5]]*SumOver[Index[Generation, 5], 3])/6 + 
   (g2^2*T^2*cHq3[Index[Generation, 5], Index[Generation, 5]]*SumOver[Index[Generation, 5], 3]*SumOver[SUNFIndex[Col5], 3])/6, 
lmbdH2B0W0 -> (cHbox*g1*g2*T^2)/4 + (5*cHD*g1*g2*T^2)/24 - (g1*g2*T^2*cHe[Index[Generation, 5], Index[Generation, 5]]*SumOver[Index[Generation, 5], 3])/6 - (g1*g2*T^2*cHl1[Index[Generation, 5], Index[Generation, 5]]*SumOver[Index[Generation, 5], 3])/
    6 + (g1*g2*T^2*cHl3[Index[Generation, 5], Index[Generation, 5]]*SumOver[Index[Generation, 5], 3])/6 - (g1*g2*T^2*cHd[Index[Generation, 5], Index[Generation, 5]]*SumOver[Index[Generation, 5], 3]*SumOver[SUNFIndex[Col5], 3])/18 + 
   (g1*g2*T^2*cHq1[Index[Generation, 5], Index[Generation, 5]]*SumOver[Index[Generation, 5], 3]*SumOver[SUNFIndex[Col5], 3])/18 + (g1*g2*T^2*cHq3[Index[Generation, 5], Index[Generation, 5]]*SumOver[Index[Generation, 5], 3]*SumOver[SUNFIndex[Col5], 3])/
    6 + (g1*g2*T^2*cHu[Index[Generation, 5], Index[Generation, 5]]*SumOver[Index[Generation, 5], 3]*SumOver[SUNFIndex[Col5], 3])/9, 
lmbdW04 -> 0, 
lmbdB02W02 -> 0}

(*** Two-loop matching (scalar masses) ***)

TwoLoopMatching = {mH2 -> -1/4*(cH*T^4) + (7*cHB*g1^2*T^4)/72 + (cHbox*g1^2*T^4)/144 + (13*cHD*g1^2*T^4)/576 + (cHWB*g1*g2*T^4)/16 + (cHbox*g2^2*T^4)/48 + (cHD*g2^2*T^4)/64 + (5*cHW*g2^2*T^4)/12 + (3*c3W*g2^3*T^4)/16 + (cHG*g3^2*T^4)/2 + (cHbox*lmbd*T^4)/3 - 
   (cHD*lmbd*T^4)/8 + (25*cHB*g1^2*T^4*SumOver[Index[Generation, 3], 3])/216 + (5*cHW*g2^2*T^4*SumOver[Index[Generation, 3], 3])/24 + 5*cHG*g3^2*T^4*SumOver[Index[Generation, 3], 3] - 
   (5*g1^2*T^4*cHd[Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3])/288 - (5*g1^2*T^4*cHe[Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3])/288 - 
   (5*g1^2*T^4*cHl1[Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3])/288 + (5*g2^2*T^4*cHl3[Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3])/96 + 
   (5*g1^2*T^4*cHq1[Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3])/288 + (5*g2^2*T^4*cHq3[Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3])/32 + 
   (5*g1^2*T^4*cHu[Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3])/144 + (g1*T^4*cdB[Index[Generation, 3], Index[Generation, 4]]*Conjugate[YD[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*
     SumOver[Index[Generation, 4], 3])/192 - (9*T^4*cdH[Index[Generation, 3], Index[Generation, 4]]*Conjugate[YD[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/64 + 
   (3*g2*T^4*cdW[Index[Generation, 3], Index[Generation, 4]]*Conjugate[YD[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/64 - 
   (g1*T^4*cdB[Index[Generation, 4], Index[Generation, 3]]*Conjugate[YD[Index[Generation, 4], Index[Generation, 3]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/96 - 
   (g1*T^4*ceB[Index[Generation, 3], Index[Generation, 4]]*Conjugate[YE[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/192 - 
   (3*T^4*ceH[Index[Generation, 3], Index[Generation, 4]]*Conjugate[YE[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/64 + 
   (g2*T^4*ceW[Index[Generation, 3], Index[Generation, 4]]*Conjugate[YE[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/64 - 
   (g1*T^4*ceB[Index[Generation, 4], Index[Generation, 3]]*Conjugate[YE[Index[Generation, 4], Index[Generation, 3]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/96 + 
   (g1*T^4*Conjugate[YU[Index[Generation, 3], Index[Generation, 4]]]*cuB[Index[Generation, 3], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/192 + 
   (g1*T^4*Conjugate[YU[Index[Generation, 4], Index[Generation, 3]]]*cuB[Index[Generation, 4], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/48 + 
   (3*g3*T^4*Conjugate[YU[Index[Generation, 3], Index[Generation, 4]]]*cuG[Index[Generation, 3], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/8 + 
   (3*g3*T^4*Conjugate[YU[Index[Generation, 4], Index[Generation, 3]]]*cuG[Index[Generation, 4], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/8 - 
   (9*T^4*Conjugate[YU[Index[Generation, 3], Index[Generation, 4]]]*cuH[Index[Generation, 3], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/64 + 
   (3*g2*T^4*Conjugate[YU[Index[Generation, 3], Index[Generation, 4]]]*cuW[Index[Generation, 3], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/64 + 
   (T^4*clequ1[Index[Generation, 5], Index[Generation, 6], Index[Generation, 3], Index[Generation, 4]]*Conjugate[YE[Index[Generation, 5], Index[Generation, 6]]]*Conjugate[YU[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*
     SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*SumOver[Index[Generation, 6], 3])/48 - (T^4*Conjugate[YD[Index[Generation, 5], Index[Generation, 6]]]*Conjugate[YU[Index[Generation, 3], Index[Generation, 4]]]*
     cquqd1[Index[Generation, 3], Index[Generation, 4], Index[Generation, 5], Index[Generation, 6]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*SumOver[Index[Generation, 6], 3])/16 - 
   (T^4*Conjugate[YD[Index[Generation, 5], Index[Generation, 6]]]*Conjugate[YU[Index[Generation, 3], Index[Generation, 4]]]*cquqd1[Index[Generation, 5], Index[Generation, 4], Index[Generation, 3], Index[Generation, 6]]*SumOver[Index[Generation, 3], 3]*
     SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*SumOver[Index[Generation, 6], 3])/96 - (T^4*Conjugate[YD[Index[Generation, 5], Index[Generation, 6]]]*Conjugate[YU[Index[Generation, 3], Index[Generation, 4]]]*
     cquqd8[Index[Generation, 5], Index[Generation, 4], Index[Generation, 3], Index[Generation, 6]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*SumOver[Index[Generation, 6], 3])/72 + 
   (g1*T^4*Conjugate[cdB[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*YD[Index[Generation, 3], Index[Generation, 4]])/192 - 
   (9*T^4*Conjugate[cdH[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*YD[Index[Generation, 3], Index[Generation, 4]])/64 + 
   (3*g2*T^4*Conjugate[cdW[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*YD[Index[Generation, 3], Index[Generation, 4]])/64 + 
   (5*cHbox*T^4*Conjugate[YD[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*YD[Index[Generation, 3], Index[Generation, 4]])/96 - 
   (5*cHD*T^4*Conjugate[YD[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*YD[Index[Generation, 3], Index[Generation, 4]])/192 - 
   (T^4*cHd[Index[Generation, 4], Index[Generation, 5]]*Conjugate[YD[Index[Generation, 3], Index[Generation, 5]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*
     YD[Index[Generation, 3], Index[Generation, 4]])/192 - (T^4*cledq[Index[Generation, 5], Index[Generation, 6], Index[Generation, 4], Index[Generation, 3]]*Conjugate[YE[Index[Generation, 5], Index[Generation, 6]]]*SumOver[Index[Generation, 3], 3]*
     SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*SumOver[Index[Generation, 6], 3]*YD[Index[Generation, 3], Index[Generation, 4]])/48 + 
   (T^4*Conjugate[YD[Index[Generation, 5], Index[Generation, 6]]]*cqd1[Index[Generation, 5], Index[Generation, 3], Index[Generation, 4], Index[Generation, 6]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*
     SumOver[Index[Generation, 5], 3]*SumOver[Index[Generation, 6], 3]*YD[Index[Generation, 3], Index[Generation, 4]])/24 + 
   (T^4*Conjugate[YD[Index[Generation, 5], Index[Generation, 6]]]*cqd8[Index[Generation, 5], Index[Generation, 3], Index[Generation, 4], Index[Generation, 6]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*
     SumOver[Index[Generation, 5], 3]*SumOver[Index[Generation, 6], 3]*YD[Index[Generation, 3], Index[Generation, 4]])/18 - 
   (T^4*cHd[Index[Generation, 5], Index[Generation, 4]]*Conjugate[YD[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*
     YD[Index[Generation, 3], Index[Generation, 5]])/192 - (g1*T^4*Conjugate[cdB[Index[Generation, 4], Index[Generation, 3]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*YD[Index[Generation, 4], Index[Generation, 3]])/96 + 
   (T^4*cHq1[Index[Generation, 5], Index[Generation, 4]]*Conjugate[YD[Index[Generation, 5], Index[Generation, 3]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*
     YD[Index[Generation, 4], Index[Generation, 3]])/192 + (T^4*cHq3[Index[Generation, 5], Index[Generation, 4]]*Conjugate[YD[Index[Generation, 5], Index[Generation, 3]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*
     SumOver[Index[Generation, 5], 3]*YD[Index[Generation, 4], Index[Generation, 3]])/64 + (T^4*cHq1[Index[Generation, 4], Index[Generation, 5]]*Conjugate[YD[Index[Generation, 4], Index[Generation, 3]]]*SumOver[Index[Generation, 3], 3]*
     SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*YD[Index[Generation, 5], Index[Generation, 3]])/192 + 
   (T^4*cHq3[Index[Generation, 4], Index[Generation, 5]]*Conjugate[YD[Index[Generation, 4], Index[Generation, 3]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*
     YD[Index[Generation, 5], Index[Generation, 3]])/64 - (g1*T^4*Conjugate[ceB[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*YE[Index[Generation, 3], Index[Generation, 4]])/192 - 
   (3*T^4*Conjugate[ceH[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*YE[Index[Generation, 3], Index[Generation, 4]])/64 + 
   (g2*T^4*Conjugate[ceW[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*YE[Index[Generation, 3], Index[Generation, 4]])/64 + 
   (5*cHbox*T^4*Conjugate[YE[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*YE[Index[Generation, 3], Index[Generation, 4]])/288 - 
   (5*cHD*T^4*Conjugate[YE[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*YE[Index[Generation, 3], Index[Generation, 4]])/576 - 
   (T^4*cHe[Index[Generation, 4], Index[Generation, 5]]*Conjugate[YE[Index[Generation, 3], Index[Generation, 5]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*
     YE[Index[Generation, 3], Index[Generation, 4]])/576 - (T^4*Conjugate[cledq[Index[Generation, 3], Index[Generation, 4], Index[Generation, 6], Index[Generation, 5]]]*Conjugate[YD[Index[Generation, 5], Index[Generation, 6]]]*
     SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*SumOver[Index[Generation, 6], 3]*YE[Index[Generation, 3], Index[Generation, 4]])/48 + 
   (T^4*cle[Index[Generation, 5], Index[Generation, 3], Index[Generation, 4], Index[Generation, 6]]*Conjugate[YE[Index[Generation, 5], Index[Generation, 6]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*
     SumOver[Index[Generation, 5], 3]*SumOver[Index[Generation, 6], 3]*YE[Index[Generation, 3], Index[Generation, 4]])/72 - 
   (T^4*cHe[Index[Generation, 5], Index[Generation, 4]]*Conjugate[YE[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*
     YE[Index[Generation, 3], Index[Generation, 5]])/576 - (g1*T^4*Conjugate[ceB[Index[Generation, 4], Index[Generation, 3]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*YE[Index[Generation, 4], Index[Generation, 3]])/96 + 
   (T^4*cHl1[Index[Generation, 5], Index[Generation, 4]]*Conjugate[YE[Index[Generation, 5], Index[Generation, 3]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*
     YE[Index[Generation, 4], Index[Generation, 3]])/576 + (T^4*cHl3[Index[Generation, 5], Index[Generation, 4]]*Conjugate[YE[Index[Generation, 5], Index[Generation, 3]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*
     SumOver[Index[Generation, 5], 3]*YE[Index[Generation, 4], Index[Generation, 3]])/192 + (T^4*cHl1[Index[Generation, 4], Index[Generation, 5]]*Conjugate[YE[Index[Generation, 4], Index[Generation, 3]]]*SumOver[Index[Generation, 3], 3]*
     SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*YE[Index[Generation, 5], Index[Generation, 3]])/576 + 
   (T^4*cHl3[Index[Generation, 4], Index[Generation, 5]]*Conjugate[YE[Index[Generation, 4], Index[Generation, 3]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*
     YE[Index[Generation, 5], Index[Generation, 3]])/192 + (g1*T^4*Conjugate[cuB[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*YU[Index[Generation, 3], Index[Generation, 4]])/192 + 
   (3*g3*T^4*Conjugate[cuG[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*YU[Index[Generation, 3], Index[Generation, 4]])/8 - 
   (9*T^4*Conjugate[cuH[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*YU[Index[Generation, 3], Index[Generation, 4]])/64 + 
   (3*g2*T^4*Conjugate[cuW[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*YU[Index[Generation, 3], Index[Generation, 4]])/64 + 
   (5*cHbox*T^4*Conjugate[YU[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*YU[Index[Generation, 3], Index[Generation, 4]])/96 - 
   (5*cHD*T^4*Conjugate[YU[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*YU[Index[Generation, 3], Index[Generation, 4]])/192 - 
   (T^4*cHud[Index[Generation, 4], Index[Generation, 5]]*Conjugate[YD[Index[Generation, 3], Index[Generation, 5]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*
     YU[Index[Generation, 3], Index[Generation, 4]])/192 + (T^4*cHu[Index[Generation, 4], Index[Generation, 5]]*Conjugate[YU[Index[Generation, 3], Index[Generation, 5]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*
     SumOver[Index[Generation, 5], 3]*YU[Index[Generation, 3], Index[Generation, 4]])/192 - (T^4*cHud[Index[Generation, 5], Index[Generation, 4]]*Conjugate[YD[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*
     SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*YU[Index[Generation, 3], Index[Generation, 5]])/192 + 
   (T^4*cHu[Index[Generation, 5], Index[Generation, 4]]*Conjugate[YU[Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*
     YU[Index[Generation, 3], Index[Generation, 5]])/192 + (g1*T^4*Conjugate[cuB[Index[Generation, 4], Index[Generation, 3]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*YU[Index[Generation, 4], Index[Generation, 3]])/48 + 
   (3*g3*T^4*Conjugate[cuG[Index[Generation, 4], Index[Generation, 3]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*YU[Index[Generation, 4], Index[Generation, 3]])/8 - 
   (T^4*cHq1[Index[Generation, 5], Index[Generation, 4]]*Conjugate[YU[Index[Generation, 5], Index[Generation, 3]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*
     YU[Index[Generation, 4], Index[Generation, 3]])/192 + (T^4*cHq3[Index[Generation, 5], Index[Generation, 4]]*Conjugate[YU[Index[Generation, 5], Index[Generation, 3]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*
     SumOver[Index[Generation, 5], 3]*YU[Index[Generation, 4], Index[Generation, 3]])/64 - (T^4*cHq1[Index[Generation, 4], Index[Generation, 5]]*Conjugate[YU[Index[Generation, 4], Index[Generation, 3]]]*SumOver[Index[Generation, 3], 3]*
     SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*YU[Index[Generation, 5], Index[Generation, 3]])/192 + 
   (T^4*cHq3[Index[Generation, 4], Index[Generation, 5]]*Conjugate[YU[Index[Generation, 4], Index[Generation, 3]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*
     YU[Index[Generation, 5], Index[Generation, 3]])/64 + (T^4*Conjugate[YU[Index[Generation, 3], Index[Generation, 4]]]*cqu1[Index[Generation, 3], Index[Generation, 5], Index[Generation, 6], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*
     SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*SumOver[Index[Generation, 6], 3]*YU[Index[Generation, 5], Index[Generation, 6]])/24 + 
   (T^4*Conjugate[YU[Index[Generation, 3], Index[Generation, 4]]]*cqu8[Index[Generation, 3], Index[Generation, 5], Index[Generation, 6], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*
     SumOver[Index[Generation, 5], 3]*SumOver[Index[Generation, 6], 3]*YU[Index[Generation, 5], Index[Generation, 6]])/18 - 
   (T^4*Conjugate[cquqd1[Index[Generation, 3], Index[Generation, 6], Index[Generation, 5], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*SumOver[Index[Generation, 6], 3]*
     YD[Index[Generation, 3], Index[Generation, 4]]*YU[Index[Generation, 5], Index[Generation, 6]])/96 - (T^4*Conjugate[cquqd1[Index[Generation, 5], Index[Generation, 6], Index[Generation, 3], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*
     SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*SumOver[Index[Generation, 6], 3]*YD[Index[Generation, 3], Index[Generation, 4]]*YU[Index[Generation, 5], Index[Generation, 6]])/16 - 
   (T^4*Conjugate[cquqd8[Index[Generation, 3], Index[Generation, 6], Index[Generation, 5], Index[Generation, 4]]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*SumOver[Index[Generation, 6], 3]*
     YD[Index[Generation, 3], Index[Generation, 4]]*YU[Index[Generation, 5], Index[Generation, 6]])/72 + (T^4*Conjugate[clequ1[Index[Generation, 3], Index[Generation, 4], Index[Generation, 5], Index[Generation, 6]]]*SumOver[Index[Generation, 3], 3]*
     SumOver[Index[Generation, 4], 3]*SumOver[Index[Generation, 5], 3]*SumOver[Index[Generation, 6], 3]*YE[Index[Generation, 3], Index[Generation, 4]]*YU[Index[Generation, 5], Index[Generation, 6]])/48,
mB02 -> (cHbox*g1^2*T^4)/18 + (cHD*g1^2*T^4)/18 - (g1^2*T^4*cHd[Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3])/9 - (g1^2*T^4*cHe[Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3])/9 - 
   (g1^2*T^4*cHl1[Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3])/9 + (g1^2*T^4*cHq1[Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3])/9 + 
   (2*g1^2*T^4*cHu[Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3])/9 + (g1^2*T^4*cdd[Index[Generation, 3], Index[Generation, 3], Index[Generation, 4], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*
     SumOver[Index[Generation, 4], 3])/36 + (g1^2*T^4*cdd[Index[Generation, 3], Index[Generation, 4], Index[Generation, 4], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/108 + 
   (g1^2*T^4*cdd[Index[Generation, 4], Index[Generation, 3], Index[Generation, 3], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/108 + 
   (g1^2*T^4*cdd[Index[Generation, 4], Index[Generation, 4], Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/36 + 
   (g1^2*T^4*ced[Index[Generation, 3], Index[Generation, 3], Index[Generation, 4], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/36 + 
   (g1^2*T^4*ced[Index[Generation, 4], Index[Generation, 4], Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/36 + 
   (g1^2*T^4*cee[Index[Generation, 3], Index[Generation, 3], Index[Generation, 4], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/36 + 
   (g1^2*T^4*cee[Index[Generation, 3], Index[Generation, 4], Index[Generation, 4], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/36 + 
   (g1^2*T^4*cee[Index[Generation, 4], Index[Generation, 3], Index[Generation, 3], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/36 + 
   (g1^2*T^4*cee[Index[Generation, 4], Index[Generation, 4], Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/36 - 
   (g1^2*T^4*ceu[Index[Generation, 3], Index[Generation, 3], Index[Generation, 4], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/18 - 
   (g1^2*T^4*ceu[Index[Generation, 4], Index[Generation, 4], Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/18 + 
   (g1^2*T^4*cld[Index[Generation, 3], Index[Generation, 3], Index[Generation, 4], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/36 + 
   (g1^2*T^4*cld[Index[Generation, 4], Index[Generation, 4], Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/36 + 
   (g1^2*T^4*cle[Index[Generation, 3], Index[Generation, 3], Index[Generation, 4], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/36 + 
   (g1^2*T^4*cle[Index[Generation, 4], Index[Generation, 4], Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/36 + 
   (g1^2*T^4*cll[Index[Generation, 3], Index[Generation, 3], Index[Generation, 4], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/36 + 
   (g1^2*T^4*cll[Index[Generation, 3], Index[Generation, 4], Index[Generation, 4], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/72 + 
   (g1^2*T^4*cll[Index[Generation, 4], Index[Generation, 3], Index[Generation, 3], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/72 + 
   (g1^2*T^4*cll[Index[Generation, 4], Index[Generation, 4], Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/36 - 
   (g1^2*T^4*clq1[Index[Generation, 3], Index[Generation, 3], Index[Generation, 4], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/36 - 
   (g1^2*T^4*clq1[Index[Generation, 4], Index[Generation, 4], Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/36 - 
   (g1^2*T^4*clu[Index[Generation, 3], Index[Generation, 3], Index[Generation, 4], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/18 - 
   (g1^2*T^4*clu[Index[Generation, 4], Index[Generation, 4], Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/18 - 
   (g1^2*T^4*cqd1[Index[Generation, 3], Index[Generation, 3], Index[Generation, 4], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/36 - 
   (g1^2*T^4*cqd1[Index[Generation, 4], Index[Generation, 4], Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/36 - 
   (g1^2*T^4*cqe[Index[Generation, 3], Index[Generation, 3], Index[Generation, 4], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/36 - 
   (g1^2*T^4*cqe[Index[Generation, 4], Index[Generation, 4], Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/36 + 
   (g1^2*T^4*cqq1[Index[Generation, 3], Index[Generation, 3], Index[Generation, 4], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/36 + 
   (g1^2*T^4*cqq1[Index[Generation, 3], Index[Generation, 4], Index[Generation, 4], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/216 + 
   (g1^2*T^4*cqq1[Index[Generation, 4], Index[Generation, 3], Index[Generation, 3], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/216 + 
   (g1^2*T^4*cqq1[Index[Generation, 4], Index[Generation, 4], Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/36 + 
   (g1^2*T^4*cqq3[Index[Generation, 3], Index[Generation, 4], Index[Generation, 4], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/72 + 
   (g1^2*T^4*cqq3[Index[Generation, 4], Index[Generation, 3], Index[Generation, 3], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/72 + 
   (g1^2*T^4*cqu1[Index[Generation, 3], Index[Generation, 3], Index[Generation, 4], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/18 + 
   (g1^2*T^4*cqu1[Index[Generation, 4], Index[Generation, 4], Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/18 - 
   (g1^2*T^4*cud1[Index[Generation, 3], Index[Generation, 3], Index[Generation, 4], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/18 - 
   (g1^2*T^4*cud1[Index[Generation, 4], Index[Generation, 4], Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/18 + 
   (g1^2*T^4*cuu[Index[Generation, 3], Index[Generation, 3], Index[Generation, 4], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/9 + 
   (g1^2*T^4*cuu[Index[Generation, 3], Index[Generation, 4], Index[Generation, 4], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/27 + 
   (g1^2*T^4*cuu[Index[Generation, 4], Index[Generation, 3], Index[Generation, 3], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/27 + 
   (g1^2*T^4*cuu[Index[Generation, 4], Index[Generation, 4], Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/9, 
mW02 -> (cHbox*g2^2*T^4)/18 + (g2^2*T^4*cHl3[Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3])/9 + (g2^2*T^4*cHq3[Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3])/3 + 
   (g2^2*T^4*cll[Index[Generation, 3], Index[Generation, 4], Index[Generation, 4], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/72 + 
   (g2^2*T^4*cll[Index[Generation, 4], Index[Generation, 3], Index[Generation, 3], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/72 + 
   (g2^2*T^4*clq3[Index[Generation, 3], Index[Generation, 3], Index[Generation, 4], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/12 + 
   (g2^2*T^4*clq3[Index[Generation, 4], Index[Generation, 4], Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/12 + 
   (g2^2*T^4*cqq1[Index[Generation, 3], Index[Generation, 4], Index[Generation, 4], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/24 + 
   (g2^2*T^4*cqq1[Index[Generation, 4], Index[Generation, 3], Index[Generation, 3], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/24 + 
   (g2^2*T^4*cqq3[Index[Generation, 3], Index[Generation, 3], Index[Generation, 4], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/4 - 
   (g2^2*T^4*cqq3[Index[Generation, 3], Index[Generation, 4], Index[Generation, 4], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/24 - 
   (g2^2*T^4*cqq3[Index[Generation, 4], Index[Generation, 3], Index[Generation, 3], Index[Generation, 4]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/24 + 
   (g2^2*T^4*cqq3[Index[Generation, 4], Index[Generation, 4], Index[Generation, 3], Index[Generation, 3]]*SumOver[Index[Generation, 3], 3]*SumOver[Index[Generation, 4], 3])/4}
