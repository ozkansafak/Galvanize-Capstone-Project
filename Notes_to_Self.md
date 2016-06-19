

Deleting u.DS_Store from remotely on git repo.
```javascript
find . -name ".DS_Store" -exec git rm --cached -f {} \;.
git commit -m "delete files"
git push
```
