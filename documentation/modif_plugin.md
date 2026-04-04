Si modifs plugin alors dans dossier `obsidian` :
```powershell
vsce package
```
Puis :
```powershell
code --install-extension obsidian-0.0.1.vsix
```
Puis `ctrl+shift+P` et `Developer: Reload Window`