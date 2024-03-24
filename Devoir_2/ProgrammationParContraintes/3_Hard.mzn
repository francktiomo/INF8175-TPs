% Shimbi (2119138)
% Tiomo  (2093771)

include "globals.mzn";

% Énumérations des équipes participant au tournoi
enum equipes = {CO, TA, HO, DA, NA, MU, AL, UT};

% Lieux de rencontres des equipes (0 = pas de rencontre).
% Tout lieu i, tel que 0 < i <= card(equipes), 
% est designé comme le domicile de l'équipe associée par la relation to_enum(equipe, i)
set of int: terrains = 0..card(equipes);

% Étendue du tournoi (en jours)
int: d = 16;
set of int: jours = 1..d;

% Calendrier de placement des matchs
array[equipes, jours] of var terrains: calendrier;

% Nombres de rencontres entre deux equipes
array[equipes, equipes] of var 0..2: rencontres;

% Tuples des dates de match entre deux équipes
array[equipes, equipes] of var tuple (0..d, 0..d): dates;

% Variable objectif à maximiser
var int: objectif_match_espaces;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%

% Contraintes 1 et 4: Pour effectuer une rencontre, deux équipe doivent être présente sur le même terrain qui n'est pas 0.
constraint forall(e1 in equipes, j in jours)(
  calendrier[e1,j] != 0 -> exists(e2 in equipes where e2 != e1)(calendrier[e2,j] = calendrier[e1,j])
);

% Contrainte 2: Un match doit avoir lieu sur le terrain désigné comme le domicile de l’une des deux équipes.
constraint forall(e1 in equipes, e2 in equipes where e1 != e2)(
  sum(j in jours)(calendrier[e1,j] = e2 /\ calendrier[e2,j] = e2) = 1
);

% Contrainte 3: Un terrain ne peut accueillir plus de deux équipes.
constraint forall(t in terrains diff {0}, j in jours)(
  sum([if calendrier[e,j] = t then 1 else 0 endif | e in equipes]) <= 2 
);

% Contrainte 4: Voir contrainte 1.


% Contrainte 5: Une équipe n’est pas autorisée à jouer plus de 7 jours sur son propre terrain.
constraint forall(e in equipes)(
  sum(j in jours)(calendrier[e,j] = e) <= 7 /\
  sum(e2 in equipes where e2 != e, j in jours)(calendrier[e2,j] = e) <= 7
);

% Contrainte 6: Les équipes doivent jouer deux fois l’une contre l’autre. 
constraint forall(e1 in equipes, e2 in equipes where e1 != e2) (
  rencontres[e1,e2] = 2 /\ rencontres[e2,e1] = 2
);
% Contrainte 7: Les deux rencontres de deux équipes doivent être espacées de 3 jours au minimum.
constraint forall(e1, e2 in equipes where e1 < e2)(
  let {
    var set of jours: match_jours = {j | j in jours where calendrier[e1,j] = e2 \/ calendrier[e2,j] = e1}
  } in (
    card(match_jours) = 2 /\ abs(min(match_jours) - max(match_jours)) >= 3 /\
    dates[e1,e2] = (min(match_jours), max(match_jours))
  )
);

% Contrainte 8: Les deux rencontres de deux équipes doivent avoir lieu sur des terrains différents.
constraint forall(e1, e2 in equipes where e1 != e2)(
  calendrier[e1,dates[e1,e2].1] != calendrier[e1,dates[e1,e2].2] 
);


% Objectif: Maximizer le nombre de fois où deux matchs entre les mêmes équipes sont espacés par aux moins 6 jours.
var int: match_espaces = sum(e1, e2 in equipes where e1 < e2)(
    let {
        var jours: premier_match = min([j | j in jours where calendrier[e1, j] = e2 \/ calendrier[e2, j] = e1]),
        var jours: second_match = max([j | j in jours where calendrier[e1, j] = e2 \/ calendrier[e2, j] = e1])
    } in (
        if second_match - premier_match >= 6 then 1 else 0 endif
    )
);

% Résoudre en maximisant l'objectif.
constraint objectif_match_espaces = match_espaces;
solve maximize objectif_match_espaces;

%-----------------------------------------------------------------------------%

% Affichage du calendrier par equipes et par jours puis du résultat de la fonction objectif.
% Lorsqu'une équipe ne joue pas, l'emplacement de match reste vide sinon elle contient le nom de l'équipie adverse.
% Si une équipe ne joue pas à domicile, un "@" précède le nom de l'équipe adverse.
output 
["Calendrier:\n"]
++
[if j = 1 then "   " else "" endif ++ if j < 10 then "  j" ++ show(j) else " j" ++ show(j) endif ++ if j = d then "\n" else "" endif | j in jours ]
++
[if j = 1 then show(e1) ++ ":" else "" endif ++ if fix(calendrier[e1,j]) = 0 then "    " else if fix(calendrier[e1,j]) = e1 then "  " ++ concat([if e1 != e2 /\ fix(calendrier[e2,j]) = e1 then show(e2) else "" endif | e2 in equipes])  else " @" ++ show(to_enum(equipes, calendrier[e1,j])) endif endif ++ if j = d then "\n" else "" endif | e1 in equipes, j in jours ]
++
["\nMatchs retour espacés de 6 jours ou plus = " ++ show(objectif_match_espaces) ++ "\n"]
;
