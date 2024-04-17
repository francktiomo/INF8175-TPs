% Shimbi (2119138)
% Tiomo  (2093771)

include "globals.mzn";

% États possibles pour une case du terrain
% V = Vide
% B = Tour bleue
% R = Tour rouge
enum etat = {V, B, R};

% Dimension du terrain (carré)
int: n;
set of int: dim = 1..n;

% État du terrain
array[dim, dim] of var etat: terrain;

% Variable objectif à maximiser
var int: total_commun;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%

% Contrainte 1: La somme des tours bleues doit être égale à la somme des tours rouges.
constraint
  count([terrain[i, j] | i in dim, j in dim], B) = count([terrain[i, j] | i in dim, j in dim], R);

% Contrainte 2: Pas de tours rouges et bleues sur la même ligne.
constraint
  forall(i in dim)(
    sum(j in dim)(terrain[i, j] = B) + sum(j in dim)(terrain[i, j] = R) <= 1
  );

% Contrainte 3: Pas de tours rouges et bleues sur la même colonne.
constraint
  forall(j in dim)(
    sum(i in dim)(terrain[i, j] = B) + sum(i in dim)(terrain[i, j] = R) <= 1
  );
% Contrainte 4: Rupture de la symétrie par ordre lexicographique sur la première tour de chaque couleur dans chaque ligne
constraint
  forall(i in dim)(
    let {
      array[int] of var opt int: bleu_positions = [j | j in dim where terrain[i, j] = B],
      array[int] of var opt int: rouge_positions = [j | j in dim where terrain[i, j] = R]
    } in
    if length(bleu_positions) > 0 /\ length(rouge_positions) > 0 then
      bleu_positions[1] < rouge_positions[1]
    else
      true % Aucune contrainte si l'une des couleurs n'est pas présente dans la ligne.
    endif
  );

 % Contrainte 5: La diagonale ne peut pas avoir à la fois des tours rouges et des tours bleues en même temps. 
constraint
  sum(i in dim)(terrain[i, i] = B \/ terrain[i, i] = R) <= 1;
  

% Contrainte 6: L'anti-diagonale ne peut pas avoir à la fois des tours rouges et bleues.
constraint
  sum(i in dim)(terrain[i, n - i + 1] = B \/ terrain[i, n - i + 1] = R) <= 1;

% Calculer la somme totale des tours rouges et bleues.
var int: total_rouge = sum(i, j in dim) (terrain[i, j] = R);
var int: total_bleu = sum(i, j in dim) (terrain[i, j] = B);

% Définir un prédicat pour vérifier si une k-diagonal a à la fois des tours rouges et bleues.
predicate pas_les_deux_couleurs_sur_k_diagonal(array[int] of var etat: diag) =
  sum(d in diag)(d = B \/ d = R) <= 1;
% Contrainte 7: Une k-diagonale ne peut pas avoir à la fois des tours rouges et bleues.
constraint
  forall(k in -n+2..n-2)(
    pas_les_deux_couleurs_sur_k_diagonal([terrain[i, i+k] | i in dim where i+k in dim])
  );
  
% Définir un prédicat pour vérifier si une skew-diagonal a à la fois des tours rouges et bleues.
predicate pas_les_deux_couleurs_sur_skew_diagonal(array[int] of var etat: diag) =
  sum(d in diag)(d = B \/ d = R) <= 1;
% Contrainte 8: Une skew-diagonal ne peut pas avoir à la fois des tours rouges et bleues.
constraint
  forall(k in -n+2..n-2)(
    pas_les_deux_couleurs_sur_skew_diagonal([terrain[i, n-i-k+1] | i in dim where n-i-k+1 in dim])
  );

% Maximiser le nombre total de tours.
total_commun = (total_rouge + total_bleu) div 2;

solve maximize total_commun;

%-----------------------------------------------------------------------------%
output 
["Terrain :\n"]
++
[if j = 1 then "["  endif ++ if fix(terrain[i,j]) == V then "_" else show(terrain[i,j]) endif ++ if j = n then "]\n" else "," endif | i in 1..n, j in 1..n]
++
["\n"]
++
["Total de tours commun: " ++ show(total_commun)]
