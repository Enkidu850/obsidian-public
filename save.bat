@echo off

set /p commentaire="Entrez le commentaire de commit : "

echo Empaquetage de l'extension...
rem vsce package

echo Installation de l'extension dans VS Code...
rem code --install-extension obsidian-0.0.1.vsix

rem echo Préparation de l'extension pour le déploiement...
git add .

rem echo Mise à jour de l'extension dans le dépôt Git...
git commit -m %commentaire%

rem echo Pousser les modifications vers le dépôt distant...
git push -u origin main

echo Opération terminée.

pause
