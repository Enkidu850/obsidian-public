# OBSIDIAN

Langage de programmation créé par Enkidu\
Ecrit en : Python

## Utilisation

Types uniques (définis avec `#` ):
* **INT**,
* **FLOAT**,
* **STR**,
* **BOOL**
<!-- Sortie de liste -->
Types multiples (définis avec `[]` ):
* **LIST**
<!-- Sortie de liste -->
Types géométriques (définis avec `$` ):
* **POINT**,
* **LINE**,
* **POLYGON**

Déclaration d'une variable unique :
```js
INT nom_variable# <- 10;
```
Déclaration d'une variable multiple :
```js
LIST nom_variable[] <- ["Hello World!", 6, FALSE];
```
Déclaration d'une variable géométrique (code EPSG de la donnée à définir, par défaut en 4326):
```js
POINT nom_variable$ <- (48, 2) [4326];
```

Affichage d'une variable unique :
```js
PRINT variable#;
```
Affichage d'un élément d'une variable multiple :
```js
PRINT liste[]{0};
```

Affichage d'une donnée géométrique sur une carte :
```js
MAP ligne$;
```

Commentaires :
```js
{-- Je suis un commentaire --}
```

Conditions :  
_A noter : les blocs ELIF et ELSE sont facultatifs_
```js
IF (age# = 18) THEN
    PRINT "égal";
ELSEIF (age# > 18) THEN
    PRINT "supérieur";
ELSE
    PRINT "inférieur";
END
```

## Prérequis

* NodeJS (14+)
* npm
* VSCode
```powershell
npm install --save-dev @vscode/vsce
```

## Installation

### Commande d'exécution

Dans le dossier `obsidian`, installation de la commande :
```powershell
python -m pip install .
```

Puis ajoutez `obsidian.exe` au PATH : (exemple avec Python 3.12)
```powershell
$scripts_path = "$env:APPDATA\Python\Python312\Scripts"
[Environment]::SetEnvironmentVariable("PATH", "$env:PATH;$scripts_path", "User")
``` 

Si Obsidian a été installé sur un seul profil et non de façon globale, alors il se trouvera là : (exemple avec Python 3.12)
```powershell
C:\Users\nom_utilisateur\AppData\Roaming\Python\Python312\Scripts
```

### Extension VSCode

Le projet bénéficie d'une extension VSCode permettant d'apporter de la couleur au text des fichiers `.obs` et de leur ajouter une icône.

Pour installer l'extension, dans le dossier obsidian :
```powershell
code --install-extension obsidian-0.0.1.vsix
```
Si la commande `code` ouvre des fenêtres VSCode au lieu d'exécuter l'installation, alors installez avec :
```powershell
& "$env:LOCALAPPDATA\Programs\Microsoft VS Code\bin\code.cmd" --install-extension obsidian-0.0.1.vsix
```
Dans ces cas là, `code` pointe sur `C:\Users\nom_utilisateur\AppData\Local\Programs\Microsoft VS Code\Code.exe` au lieu de `C:\Users\nom_utilisateur\AppData\Local\Programs\Microsoft VS Code\bin\code.cmd`\
Afin de régler ce problème, dans une fenêtre Powershell administrateur, exécutez :
```powershell
$profilePath = $PROFILE
if (-not (Test-Path $profilePath)) { New-Item -ItemType File -Force -Path $profilePath }
Add-Content -Path $profilePath -Value 'Set-Alias code "$env:LOCALAPPDATA\Programs\Microsoft VS Code\bin\code.cmd"'
```
Puis redémarrez le terminal et installez l'extension via la commande `code`.\
Si ça ne marche toujours pas, force.

## Exécution

Pour exécuter un fichier `.obs` contenant votre code Obsidian : (exemple sur `hello_world.obs`)
```powershell
obsidian hello_world.obs
```

