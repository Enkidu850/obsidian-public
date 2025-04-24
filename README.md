## OBSIDIAN

Langage de programmation créé par Enkidu\
Ecrit en : Python

Types uniques (définis avec `#` ): **INT**, **FLOAT**, **STR**, **BOOL**\
Types multiples (définis avec `[]` ): **LIST**\
Types géométriques (définis avec `$` ): **POINT**, **LINE**, **POLYGON**

Déclaration d'une variable unique : ```INT nom_variable# <- 10;```\
Déclaration d'une variable multiple : ```LIST nom_variable[] <- ["Hello World!", 6, FALSE];```\
Déclaration d'une variable géométrique (code EPSG de la donnée à définir, par défaut en 4326): ```POINT nom_variable$ <- (48, 2) [4326];```

Affichage d'une variable unique : ```PRINT variable#;```\
Affichage d'un élément d'une variable multiple : ```PRINT liste[]{0};```

Affichage d'une donnée géométrique sur une carte : ```MAP ligne$;```